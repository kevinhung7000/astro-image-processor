"""
星野/銀河疊圖後製腳本
====================
功能: 去光害漸層(background extraction) -> 色偏校正(色彩平衡) ->
      非線性拉伸(arcsinh, 拉出暗部細節) -> 飽和度提升 -> Clarity/銳化 -> 降噪

使用方式:
    python process_astro.py input.tif output_name

或直接修改下方 "===== 參數設定 =====" 區塊後執行:
    python process_astro.py

需要套件:
    pip install tifffile opencv-python-headless scipy numpy --break-system-packages
    pip install rawpy --break-system-packages
"""

import sys
import os
import numpy as np
import cv2
import tifffile
import rawpy
from scipy.ndimage import gaussian_filter, minimum_filter

# OpenCV 的模糊/形態學/雙邊濾波/inpaint 等運算本身就有內建多執行緒平行化,
# 這裡明確指定使用全部 CPU 核心(預設通常已是如此,顯式設定避免被其他套件覆蓋)
cv2.setNumThreads(os.cpu_count())

# ============================================================
# ===================== 參數設定(依圖調整) =====================
# ============================================================

# --- 檔案路徑 ---
INPUT_PATH = sys.argv[1] if len(sys.argv) > 1 else "input.tif"
OUTPUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "processed"
OUTPUT_DIR = "/mnt/user-data/outputs"

# --- 1. 背景漸層去除(去朦朧/光害) ---
BG_DOWNSCALE = 0.06        # 縮圖比例做背景估算,圖越大可以設更小(0.03~0.1)加速
BG_MIN_FILTER_SIZE = 9     # 抓局部最暗值的視窗大小,雜訊多可以加大(7~15)
BG_BLUR_SIGMA = 6          # 背景平滑程度,漸層變化劇烈可加大
BG_SUBTRACT_STRENGTH = 0.92  # 扣掉多少背景漸層(0~1)。太靠近1容易死黑,太小去不掉朦朧感

# --- 2. 色偏校正(中和光害顏色) ---
WB_ENABLE = True
WB_GAIN_MIN = 0.6          # 白平衡增益的裁切下限,避免過度校正
WB_GAIN_MAX = 1.8          # 白平衡增益的裁切上限

# --- 3. 非線性拉伸(拉出銀河/星雲暗部細節) ---
BLACK_POINT_PERCENTILE = 0.2   # 黑點百分位,雜訊多可以設高一點(0.5~1)先切掉底噪
STRETCH_FACTOR = 12.0          # arcsinh拉伸強度,數字越大暗部拉得越猛烈(常用範圍 5~30)
WHITE_POINT_PERCENTILE = 99.7  # 白點百分位,亮星多/有月光可以降低(99~99.9)避免死白過多

# --- 4. 飽和度 / 對比 ---
SATURATION_BOOST = 1.45   # 飽和度倍率(1.0=不變, 常用 1.2~1.6)
BRIGHTNESS_BOOST = 1.03    # 明度微調

# --- 5. Clarity(局部對比) / 銳化 ---
CLARITY_BLUR_SIGMA = 25
CLARITY_STRENGTH = 0.35
SHARPEN_BLUR_SIGMA = 2
SHARPEN_AMOUNT = 1.25      # >1 加強銳化, 1.0 = 不銳化

# --- 6. 降噪(保留星點的前提下) ---
DENOISE_ENABLE = True
DENOISE_D = 5              # 雙邊濾波視窗,雜訊嚴重可加大(但會犧牲星點銳利度)
DENOISE_SIGMA_COLOR = 15
DENOISE_SIGMA_SPACE = 15

# --- 7. 星點縮小 / 去星 ---
STAR_REDUCE_MODE = "shrink"   # "none" 不處理 / "shrink" 縮小星點 / "remove" 完全去星
STAR_DETECT_KERNEL = 5        # 偵測用的 top-hat 核大小,約略等於「一般星點的直徑(像素)」
STAR_DETECT_THRESH = 18       # top-hat 亮度門檻(0-255),雜訊多/背景不乾淨時可調高避免誤判
STAR_MAX_AREA = 250           # 一般星點的最大面積(像素),超過此面積且「形狀不圓」才視為星系核心/星雲團塊而排除
STAR_MAX_AREA_LARGE = 2500    # 允許判定為「亮星暈光」的面積上限(需同時符合下面的圓度條件,否則還是會被當成星雲排除)
STAR_CIRCULARITY_ASPECT = 1.6 # 外接框長寬比門檻,越接近1代表越圓;亮星暈光通常接近圓形,星雲/星系核心通常不規則
STAR_DILATE = 1               # 遮罩外擴的基本像素數,把星點暈光邊緣也一起處理,避免留下光暈殘邊
STAR_DILATE_SCALE = 0.15      # 依星點自身尺寸額外外擴的比例(0~1)。亮星暈光範圍大,固定 STAR_DILATE 常常不夠蓋滿,
                               # 設 >0 讓遮罩隨星點大小等比放大;設 0 則所有星點都只外擴 STAR_DILATE 個像素
STAR_SHRINK_KERNEL = 3        # shrink 模式:侵蝕核大小,越大縮得越明顯
STAR_SHRINK_ITERATIONS = 1    # shrink 模式:侵蝕次數
STAR_SHRINK_STRENGTH = 0.8    # shrink 模式:0~1,套用強度(1=完全套用侵蝕結果)
STAR_INPAINT_RADIUS = 5       # remove 模式:inpaint 取樣半徑

# --- 7b. 大範圍偵測(密集星團 / 大片暈光聚集區) ---
# 原本只有單一 STAR_DETECT_KERNEL(=5)的 top-hat,這種小核只對「單顆孤立星點」敏感;
# 星點密集聚集(例如疏散星團)在小核 top-hat 下,每顆星各自只貢獻一小塊亮區,彼此又
# 常被中間縫隙打斷,很難被判定成同一塊需要處理的區域,導致這種聚集區域去星去不乾淨。
# 這裡另外用一個大很多的 kernel 做第二次 top-hat,把整個聚集範圍當一塊「相對背景凸起」
# 抓出來,和小星點遮罩分開處理、各自用適合的 inpaint 半徑,兩者不衝突。
STAR_MULTISCALE_ENABLE = True     # 是否啟用大範圍偵測這一層(密集星團/大暈光聚集)
STAR_DETECT_KERNEL_LARGE = 21     # 大範圍偵測用的 top-hat 核大小,約略等於「一整片聚集區域」的直徑(像素)
STAR_DETECT_THRESH_LARGE = 12     # 大範圍偵測門檻(0-255),通常比單顆星點門檻低,因為整片區域對比度不如單顆星點集中
STAR_CLUSTER_MIN_AREA = 300       # 小於此面積直接交給小星點那層處理即可,避免兩層重複處理同一顆星
STAR_CLUSTER_MAX_AREA = 15000     # 大於此面積視為星雲/星系本體而非星團,予以排除,避免誤傷星雲細節
STAR_CLUSTER_ASPECT = 2.5         # 外接框長寬比門檻,比單顆亮星寬鬆(星團形狀通常不是正圓,但仍應是塊狀而非細長條的星雲絲)
STAR_CLUSTER_DILATE = 4           # 星團遮罩額外外擴像素數,把邊緣暗淡的星點也一起蓋進去
STAR_INPAINT_RADIUS_LARGE = 14    # 星團範圍 inpaint 取樣半徑,範圍大,需要比單顆星點更大的取樣半徑才能內插出合理背景

SAVE_STAR_LAYERS = True      # True 時額外輸出 _starless(去星版) 與 _starmask(星點遮罩) 供手動疊圖

# ============================================================
# ========================= 主流程 ===========================
# ============================================================

def load_image(path):
    """讀取 TIF(支援 float32 疊圖輸出 / 一般 8-16bit TIF)"""
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    # 正規化到 0~1 區間(依原始資料的最大值判斷是 8/16bit 或已是 0~1 float)
    max_val = img.max()
    if max_val > 1.5:
        img = img / (65535.0 if max_val > 255 else 255.0)
    return np.clip(img, 0, 1)


def remove_background_gradient(img):
    """估算並扣除背景漸層(光害/暈影/朦朧感)。

    三色版各自獨立扣除背景,在訊號夠強的區域(星雲核心/主體)會產生較濃郁的
    色彩分離感,這是很多人喜歡的效果,予以保留。但這個機制套在訊號極弱的
    邊緣暈影區會失控放大成色偏(例如藍邊),因為那裡三色版殘留比例本來就
    對不齊,扣得越兇偏色越明顯。

    解法: 依「局部訊號強度」算一個 0~1 的信心度權重,訊號強的地方信心度接近1、
    用原本三色版各自扣除的背景(保留濃郁色彩);訊號弱的地方信心度接近0、
    改用「亮度算一份背景、三色版等比例分配扣除」的中性版本(色彩比例不會走鐘),
    兩者依信心度平滑混合,而不是整張圖二選一。
    """
    h, w, _ = img.shape
    small_h, small_w = max(8, int(h * BG_DOWNSCALE)), max(8, int(w * BG_DOWNSCALE))

    # --- 版本A: 三色版各自獨立估算背景(保留濃郁色彩,但弱訊號區易失控) ---
    bg_full_perchannel = np.zeros_like(img)
    for c in range(3):
        ch = img[:, :, c]
        small = cv2.resize(ch, (small_w, small_h), interpolation=cv2.INTER_AREA)
        bg_small = minimum_filter(small, size=BG_MIN_FILTER_SIZE)
        bg_small = gaussian_filter(bg_small, sigma=BG_BLUR_SIGMA)
        bg_full_perchannel[:, :, c] = cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)

    # --- 版本B: 用亮度算單一份背景,三色版依原始色彩比例等比例扣除(中性、不偏色) ---
    luminance = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    small_l = cv2.resize(luminance, (small_w, small_h), interpolation=cv2.INTER_AREA)
    bg_small_l = minimum_filter(small_l, size=BG_MIN_FILTER_SIZE)
    bg_small_l = gaussian_filter(bg_small_l, sigma=BG_BLUR_SIGMA)
    bg_lum = cv2.resize(bg_small_l, (w, h), interpolation=cv2.INTER_CUBIC)
    eps = 1e-6
    ratio = img / (luminance[:, :, None] + eps)
    bg_full_neutral = bg_lum[:, :, None] * ratio

    # --- 信心度權重: 局部訊號相對於背景的凸起程度,粗尺度平滑避免雜訊化 ---
    signal_above_bg = np.clip(luminance - bg_lum, 0, None)
    confidence = gaussian_filter(signal_above_bg, sigma=BG_BLUR_SIGMA * 3)
    conf_ref = np.percentile(confidence, 97)  # 用高百分位當作「訊號很強」的基準
    confidence = np.clip(confidence / max(conf_ref, 1e-6), 0, 1) ** 0.6
    confidence = confidence[:, :, None]

    bg_full = confidence * bg_full_perchannel + (1 - confidence) * bg_full_neutral

    out = img - bg_full * BG_SUBTRACT_STRENGTH
    return np.clip(out, 0, None)


def correct_color_cast(img):
    """用暗部中位數做簡易白平衡,中和光害色偏"""
    if not WB_ENABLE:
        return img
    med = np.median(img.reshape(-1, 3), axis=0)
    target = med.mean()
    gains = target / (med + 1e-6)
    gains = np.clip(gains, WB_GAIN_MIN, WB_GAIN_MAX)
    return np.clip(img * gains[None, None, :], 0, None)


def stretch_dynamic_range(img):
    """arcsinh 非線性拉伸,拉出暗部細節同時保護亮部不死白"""
    black_point = np.percentile(img, BLACK_POINT_PERCENTILE)
    img = np.clip(img - black_point, 0, None)
    stretched = np.arcsinh(img * STRETCH_FACTOR) / np.arcsinh(STRETCH_FACTOR)
    white_point = np.percentile(stretched, WHITE_POINT_PERCENTILE)
    stretched = stretched / max(white_point, 1e-6)
    return np.clip(stretched, 0, 1)


def boost_saturation(img01):
    """飽和度 / 明度提升(在 HSV 空間操作)"""
    img8 = (img01 * 255).astype(np.uint8)
    hsv = cv2.cvtColor(img8, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURATION_BOOST, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * BRIGHTNESS_BOOST, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def apply_clarity_and_sharpen(img8):
    """局部對比(clarity) + 銳化"""
    img_f = img8.astype(np.float32)
    blur = cv2.GaussianBlur(img_f, (0, 0), sigmaX=CLARITY_BLUR_SIGMA)
    clarity = img_f + (img_f - blur) * CLARITY_STRENGTH
    clarity = np.clip(clarity, 0, 255).astype(np.uint8)

    blur2 = cv2.GaussianBlur(clarity, (0, 0), sigmaX=SHARPEN_BLUR_SIGMA)
    sharpened = cv2.addWeighted(clarity, SHARPEN_AMOUNT, blur2, 1 - SHARPEN_AMOUNT, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def denoise(img8):
    if not DENOISE_ENABLE:
        return img8
    return cv2.bilateralFilter(img8, d=DENOISE_D,
                                sigmaColor=DENOISE_SIGMA_COLOR,
                                sigmaSpace=DENOISE_SIGMA_SPACE)


def detect_star_mask(img01):
    """用 top-hat 抓出小而亮的圓形結構(星點),並用「面積 + 圓度」雙重過濾排除星系核心等大範圍亮區

    原本只用面積過濾(area <= STAR_MAX_AREA)會有個問題: 亮星本身的星芒/暈光經過 top-hat
    後,連通區域面積常常也會超過門檻,導致亮星被誤判成「像星雲核心一樣的大結構」而整顆被排除在
    遮罩之外,結果就是小星點都被正常去除,只剩下幾顆最亮的星留在 starless 版本裡。
    這裡改成: 面積在 STAR_MAX_AREA 以內一律視為星點; 面積超過但落在 STAR_MAX_AREA_LARGE 以內、
    且外接框長寬比(圓度)夠圓的,也視為亮星暈光而保留; 真正不規則的星雲/星系核心才會被排除。
    """
    gray = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (STAR_DETECT_KERNEL, STAR_DETECT_KERNEL))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(gray, opened)
    _, raw_mask = cv2.threshold(tophat, STAR_DETECT_THRESH, 255, cv2.THRESH_BINARY)

    # 用查表向量化取代逐一走訪每個連通元件,星點數量多時(數千顆)速度差異可達數十倍
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    long_side = np.maximum(widths, heights)
    short_side = np.maximum(np.minimum(widths, heights), 1)
    aspect = long_side / short_side

    small_star = areas <= STAR_MAX_AREA
    bright_halo = (areas > STAR_MAX_AREA) & (areas <= STAR_MAX_AREA_LARGE) & (aspect <= STAR_CIRCULARITY_ASPECT)
    valid = small_star | bright_halo
    valid[0] = False  # label 0 是背景,一定要排除

    mask = (valid[labels].astype(np.uint8)) * 255

    # 依每個星點自身大小做外擴:固定的 STAR_DILATE 對小星點夠用,但亮星暈光範圍大,
    # 固定外擴常常蓋不滿,做完 inpaint/縮星後邊緣會留下一圈殘影。
    # 用「依外擴量分桶」取代逐星點迴圈 dilate,桶數通常只有幾到十幾種,效能上仍是向量化等級。
    if STAR_DILATE > 0 or STAR_DILATE_SCALE > 0:
        dilate_amounts = np.zeros(num_labels, dtype=np.int32)
        dilate_amounts[valid] = np.maximum(
            1, np.round(STAR_DILATE + long_side[valid] * STAR_DILATE_SCALE).astype(np.int32)
        )
        mask_out = np.zeros_like(mask)
        for amt in np.unique(dilate_amounts[valid]):
            bucket_labels = valid & (dilate_amounts == amt)
            sub_mask = (bucket_labels[labels].astype(np.uint8)) * 255
            dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (amt * 2 + 1, amt * 2 + 1))
            sub_mask = cv2.dilate(sub_mask, dilate_kernel)
            mask_out = cv2.bitwise_or(mask_out, sub_mask)
        mask = mask_out

    return mask


def detect_cluster_mask(img01, exclude_mask=None):
    """用大 kernel 的 top-hat 抓出「整片相對背景凸起」的區域,用來補抓密集星團/大片暈光聚集區。

    小核 top-hat(detect_star_mask)只對單顆孤立星點敏感,星點彼此靠得很近時,中間縫隙會
    把整片區域切成很多小塊,面積都不大,很容易被小星點邏輯正確處理掉個別星點,但縫隙间的
    背景仍帶著整片區域偏亮的底,看起來就是「星點是去了,但整片還是霧霧亮亮的沒清乾淨」。
    用大 kernel 重做一次 top-hat,background(opening)是用大範圍估計,整片聚集區相對於
    更大範圍的背景仍然是凸起的,因此會被抓成一塊連通區域,面積夠大也夠塊狀的話就當作
    「星團/大範圍聚集區」,交給更大的 inpaint 半徑去處理。
    """
    gray = cv2.cvtColor((img01 * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (STAR_DETECT_KERNEL_LARGE, STAR_DETECT_KERNEL_LARGE))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
    tophat = cv2.subtract(gray, opened)
    _, raw_mask = cv2.threshold(tophat, STAR_DETECT_THRESH_LARGE, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA]
    widths = stats[:, cv2.CC_STAT_WIDTH]
    heights = stats[:, cv2.CC_STAT_HEIGHT]
    long_side = np.maximum(widths, heights)
    short_side = np.maximum(np.minimum(widths, heights), 1)
    aspect = long_side / short_side

    valid = (areas >= STAR_CLUSTER_MIN_AREA) & (areas <= STAR_CLUSTER_MAX_AREA) & \
            (aspect <= STAR_CLUSTER_ASPECT)
    valid[0] = False  # label 0 是背景

    mask = (valid[labels].astype(np.uint8)) * 255

    if STAR_CLUSTER_DILATE > 0:
        dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (STAR_CLUSTER_DILATE * 2 + 1, STAR_CLUSTER_DILATE * 2 + 1))
        mask = cv2.dilate(mask, dilate_kernel)

    if exclude_mask is not None:
        # 已經被小星點遮罩蓋到的核心不用重複算進星團遮罩,避免兩層互相打架
        mask = cv2.bitwise_and(mask, cv2.bitwise_not(exclude_mask))

    return mask


def shrink_stars(img01, mask):
    """只在星點遮罩範圍內做侵蝕,讓星點半徑縮小,強度可調(0~1 混合)"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (STAR_SHRINK_KERNEL, STAR_SHRINK_KERNEL))
    eroded = img01.copy()
    for c in range(3):
        eroded[:, :, c] = cv2.erode(img01[:, :, c], kernel, iterations=STAR_SHRINK_ITERATIONS)

    mask_f = (mask.astype(np.float32) / 255.0)[:, :, None] * STAR_SHRINK_STRENGTH
    out = img01 * (1 - mask_f) + eroded * mask_f
    return np.clip(out, 0, 1)


def remove_stars(img01, mask, radius=STAR_INPAINT_RADIUS):
    """用 inpaint 把星點位置用周圍背景/星雲內容內插填補,做出無星版本"""
    img8 = (img01 * 255).astype(np.uint8)
    inpainted = cv2.inpaint(img8, mask, radius, cv2.INPAINT_TELEA)
    return inpainted.astype(np.float32) / 255.0


def remove_stars_multiscale(img01, star_mask, cluster_mask):
    """分兩階段 inpaint: 先用小半徑處理單顆星點,再用大半徑處理密集星團/大片聚集區。

    先處理小星點的好處是,等到處理星團範圍時,周圍可供取樣的背景已經先被清乾淨過一輪,
    inpaint 取樣到的參考像素比較不會是「旁邊也還有星點沒去掉」的髒背景,結果會更自然。
    """
    out = img01
    if star_mask is not None and np.any(star_mask):
        out = remove_stars(out, star_mask, radius=STAR_INPAINT_RADIUS)
    if cluster_mask is not None and np.any(cluster_mask):
        out = remove_stars(out, cluster_mask, radius=STAR_INPAINT_RADIUS_LARGE)
    return out


def process_stars(img01, star_mask, cluster_mask):
    """依 STAR_REDUCE_MODE,用已算好的遮罩執行縮小或去星(星團遮罩只在 remove 模式下使用,
    shrink 模式的侵蝕對大片聚集區意義不大,維持只處理單顆星點遮罩)"""
    if STAR_REDUCE_MODE == "shrink":
        return shrink_stars(img01, star_mask)
    elif STAR_REDUCE_MODE == "remove":
        return remove_stars_multiscale(img01, star_mask, cluster_mask)
    elif STAR_REDUCE_MODE == "none":
        return img01
    else:
        print(f"  警告: 未知的 STAR_REDUCE_MODE '{STAR_REDUCE_MODE}',略過此步驟")
        return img01


def finish_pipeline(img01):
    """星點處理完成後的共用後製步驟: 飽和度/明度 -> Clarity+銳化 -> 降噪,回傳 8bit 影像"""
    img8 = boost_saturation(img01)
    img8 = apply_clarity_and_sharpen(img8)
    img8 = denoise(img8)
    return img8


def save_image(img8, name):
    out_jpg = os.path.join(OUTPUT_DIR, f"{name}.jpg")
    out_tif = os.path.join(OUTPUT_DIR, f"{name}.tif")
    cv2.imwrite(out_jpg, cv2.cvtColor(img8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 96])
    img16 = (img8.astype(np.uint16) * 257)
    cv2.imwrite(out_tif, cv2.cvtColor(img16, cv2.COLOR_RGB2BGR))
    return out_jpg, out_tif


def main():
    print(f"讀取: {INPUT_PATH}")
    img = load_image(INPUT_PATH)
    print("尺寸:", img.shape)

    print("[1/7] 去背景漸層(去朦朧/光害)...")
    img = remove_background_gradient(img)

    print("[2/7] 色偏校正...")
    img = correct_color_cast(img)

    print("[3/7] 非線性拉伸(arcsinh)...")
    img = stretch_dynamic_range(img)
    pre_star_img = img  # 保留星點處理前的版本,供輸出星點遮罩/去星背景層用

    need_mask = STAR_REDUCE_MODE != "none" or SAVE_STAR_LAYERS
    star_mask = detect_star_mask(pre_star_img) if need_mask else None
    cluster_mask = None
    if need_mask and STAR_MULTISCALE_ENABLE:
        cluster_mask = detect_cluster_mask(pre_star_img, exclude_mask=star_mask)

    print(f"[4/7] 星點處理(模式: {STAR_REDUCE_MODE})...")
    if star_mask is not None:
        img = process_stars(pre_star_img, star_mask, cluster_mask)

    print("[5/7] 飽和度/明度提升...")
    print("[6/7] Clarity + 銳化...")
    print("[7/7] 降噪...")
    img8 = finish_pipeline(img)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_jpg, out_tif = save_image(img8, OUTPUT_NAME)
    print(f"完成! 輸出:\n  {out_jpg}\n  {out_tif}")

    if SAVE_STAR_LAYERS and star_mask is not None:
        combined_mask = star_mask
        if cluster_mask is not None:
            combined_mask = cv2.bitwise_or(star_mask, cluster_mask)
        out_mask = os.path.join(OUTPUT_DIR, f"{OUTPUT_NAME}_starmask.png")
        cv2.imwrite(out_mask, combined_mask)
        print(f"  星點遮罩(含星團範圍): {out_mask}")

        # 不管主模式是 none/shrink/remove,都額外輸出一份完全去星、只留背景/星雲的版本
        if STAR_REDUCE_MODE == "remove":
            starless_img = img  # 主結果本身就已經是去星版本,不用重算
        else:
            starless_img = remove_stars_multiscale(pre_star_img, star_mask, cluster_mask)
        starless_img8 = finish_pipeline(starless_img)
        out_jpg2, out_tif2 = save_image(starless_img8, f"{OUTPUT_NAME}_starless")
        print(f"  去星背景層: {out_jpg2}\n  去星背景層: {out_tif2}")
        print("  (三張圖可在 Photoshop/PixInsight 疊圖: 主結果 + 去星背景 + 星點遮罩當選取範圍)")


if __name__ == "__main__":
    main()