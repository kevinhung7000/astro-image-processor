# **AstroImageProcessor-Py 🌌**

[English](#bookmark=id.bbg3j5m8q4vy) | [繁體中文](#bookmark=id.hp0dy39mkqri)

## ---

**English**

An automated astrophotography post-processing script tailored for wide-field Milky Way and deep-sky nebula imaging. Built on top of OpenCV, SciPy, and NumPy, this script provides a one-click pipeline to transform stacked linear TIFF files through background extraction, color calibration, robust arcsinh stretching, star control, and advanced denoising.

### **✨ Key Features**

* **Background Gradient Extraction**: Utilizes a dual-version smooth blending mechanism (per-channel independent vs. luminance neutral). This preserves rich colors in high-signal nebula cores while effectively suppressing edge color casts in low-signal areas.  
* **Color Cast Calibration**: Automatically neutralizes light pollution and balances color casts based on the median values of dark backgrounds.  
* **Non-linear Stretching (arcsinh)**：Applies an arcsinh function for aggressive dynamic range expansion. This brings out faint, dim nebulosity from the dark background while preventing bright star cores from blowing out.  
* **Multi-scale Star Control**:  
  * **Shrink Mode**: Leverages morphological erosion to reduce star sizes, making nebulae and Milky Way structures pop.  
  * **Remove Mode**: Combines small and large dual-scale Top-hat detection to cleanly isolate individual stars and dense star clusters, utilizing Inpaint algorithms to generate high-quality "Starless" backgrounds.  
* **Detail & Denoise Enhancement**: Integrates local contrast adjustment (Clarity), bilateral filtering for noise reduction, and unsharp masking.

### **📦 Requirements & Installation**

The script natively supports standard 8/16-bit TIFF files as well as 32-bit float stack outputs.  
pip install tifffile opencv-python-headless scipy numpy  
*(Optional) If you wish to read raw files directly from your camera (.CR2, .NEF, .ARW, etc.), please install rawpy as well.*

### **🚀 Usage**

#### **1\. Command Line Interface (CLI)**

Pass the input image path and the desired output name directly into the terminal:  
python process\_astro\_v2.py input.tif processed\_result

#### **2\. Manual Configuration**

Alternatively, open the script in any editor, tweak the parameters under the \===== 參數設定 \===== block to match your specific target, and run:  
python process\_astro\_v2.py

### **🔧 Target-Specific Tuning Guide**

By adjusting the parameter blocks, this script can be optimized for either wide-field Milky Way vistas or faint deep-sky nebulae:

| Parameter | Milky Way Mode (Default) | Deep-Sky Nebula Mode (Recommended) | Tuning Logic   |
| :---- | :---- | :---- | :---- |
| STRETCH\_FACTOR | 12.0 | 25.0 \~ 40.0 | Nebulae signals are incredibly faint; a much higher arcsinh stretch is required to pull out details. |
| BLACK\_POINT\_PERCENTILE | 0.2 | 0.05 \~ 0.12 | Lowering the black point preserves dim gas and dark nebula structures from being clipped. |
| BG\_SUBTRACT\_STRENGTH | 0.92 | 0.80 \~ 0.85 | A more conservative value prevents large-scale, diffuse nebulosity from being mistaken for light pollution gradients. |
| STAR\_REDUCE\_MODE | "shrink" | "shrink" or "remove" | Minimizing or eliminating overwhelming starfields allows the main nebula structure to stand out. |
| SATURATION\_BOOST | 1.45 | 1.65 \~ 1.75 | Heightens saturation to reveal vibrant H-alpha (red) and OIII (blue-green) gaseous emissions. |

### **🖼️ Processing Example**

#### **Target: NGC 2024 Flame Nebula**

* **Total Integration Time**: 1 Hour 42 Minutes  
* **Tuning Strategy**: Cranked STRETCH\_FACTOR up to 28.0 and carefully balanced BLACK\_POINT\_PERCENTILE at 0.12 to drop the light-polluted grey background. This successfully carved out the intricate fiery lanes and dust structures of the Flame Nebula.

*(Feel free to upload your before/after images here\!)*

* **Before**: (Your original dark/un-stretched image)  
* **After**: (Your successfully processed image showcasing the Flame Nebula)

### **📂 Auto-Generated Outputs**

When SAVE\_STAR\_LAYERS \= True is enabled, the script automatically outputs three files into your destination folder:

1. \[Name\].jpg / .tif: The final combined post-processed master image.  
2. \[Name\]\_starless.jpg / .tif: A completely star-free image showcasing only the nebulosity background.  
3. \[Name\]\_starmask.png: A black-and-white mask containing detected stars and clusters.

**💡 Advanced Workflow**: Import these layers into Photoshop or PixInsight. Place the *starless* layer at the bottom for independent contrast/nebula adjustments, then use the \*starmask\* to isolate and blend the stars back in. This grants you total control over the star-to-background brightness ratio\!

## ---

**繁體中文**

一個專為星野攝影、銀河與深空星雲設計的自動化後製影像處理腳本。基於 OpenCV、SciPy 與 NumPy 構建，能一鍵將疊圖後的線性 TIFF 檔案進行去光害、色彩校正、 arcsinh 非線性拉伸、星點控制及降噪處理。

### **✨ 核心功能**

* **去背景漸層 (Background Extraction)**：採用雙版本平滑混合機制（三色獨立 vs 亮度中性），在保留星雲核心濃郁色彩的同時，有效壓制弱訊號區的邊緣色偏。  
* **白平衡色偏校正**：依據暗部中位數自動中和城市光害造成的偏色。  
* **非線性拉伸 (arcsinh Stretch)**：採用 arcsinh 函數進行強力的動態範圍拉伸，在逼出暗部微弱星雲氣體的同時，有效保護亮星核心不死白。  
* **多尺度星點控制 (Multi-scale Star Control)**：  
  * **Shrink 模式**：利用形態學侵蝕縮小星點，讓星雲與銀河主體更突出。  
  * **Remove 模式**：結合大小雙核心 Top-hat 偵測，能乾淨分離出單顆星點與密集星團，並透過 Inpaint 技術生成高畫質的「無星版背景（Starless）」。  
* **細節與降噪增強**：整合局部對比（Clarity）、雙邊濾波（Bilateral Filter）降噪與銳化流程。

### **📦 安裝需求**

腳本支援處理一般的 8/16-bit TIFF，以及 32-bit float 的疊圖輸出檔案。  
pip install tifffile opencv-python-headless scipy numpy  
*(選配) 如果想直接讀取相機的 RAW 檔（如 .CR2, .NEF, .ARW 等），請額外安裝 rawpy。*

### **🚀 使用方式**

#### **1\. 命令列執行**

最簡單的使用方式，直接傳入輸入檔案路徑與輸出名稱：  
python process\_astro\_v2.py input.tif processed\_result

#### **2\. 修改參數執行**

你也可以直接用編輯器打開腳本，在 \===== 參數設定 \===== 區塊中針對不同目標優化參數，隨後直接執行：  
python process\_astro\_v2.py

### **🔧 針對不同天體的參數調校指南**

本腳本透過參數組合，可完美適應「廣角銀河」與「深空星雲」兩種截然不同的場景：

| 參數功能 | 廣角銀河模式 (預設) | 深空星雲模式 (推薦) | 調整邏輯說明   |
| :---- | :---- | :---- | :---- |
| STRETCH\_FACTOR | 12.0 | 25.0 \~ 40.0 | 星雲訊號極其微弱，需要大幅提高 arcsinh 拉伸強度。 |
| BLACK\_POINT\_PERCENTILE | 0.2 | 0.05 \~ 0.12 | 調低黑點以保護暗星雲與微弱氣體細節不被切掉。 |
| BG\_SUBTRACT\_STRENGTH | 0.92\` | 0.80 \~ 0.85 | 避免大面積擴散的星雲氣體被當成光害漸層誤擦除。 |
| STAR\_REDUCE\_MODE | "shrink" | "shrink" 或 "remove" | 透過縮星或去星，防止漫天繁星淹沒了星雲主體的風采。 |
| SATURATION\_BOOST | 1.45 | 1.65 \~ 1.75 | 提升飽和度，逼出 H-alpha (紅) 與 OIII (藍綠) 的鮮豔氣體色彩。 |

### **🖼️ 處理範例**

#### **目標：NGC 2024 火焰星雲 (Flame Nebula)**

* **總曝光時間**：1 小時 42 分鐘  
* **調校重點**：大幅拉高 STRETCH\_FACTOR 至 28.0，並將 BLACK\_POINT\_PERCENTILE 控制在 0.12 壓制光害灰底，成功讓火焰星雲的細節與暗條紋立體顯現。

*(建議在此處附上你的對比圖，這會讓你的 Repo 變得超級吸引人！)*

* **處理前 (Before)**：(放之前死黑那張)  
* **處理後 (After)**：(放成功拉出來這張)

### **📂 自動輸出檔案清單**

當開啟 SAVE\_STAR\_LAYERS \= True 時，腳本會自動在輸出資料夾生成：

1. \[名稱\].jpg / .tif：最終後製完成的綜合影像。  
2. \[名稱\]\_starless.jpg / .tif：完全去除星點、只留背景與星雲氣體的影像。  
3. \[名稱\]\_starmask.png：星點與星團的黑白遮罩（Mask）。

**💡 進階玩法**：你可以將這三個檔案匯入 Photoshop 或 PixInsight 中。將「去星背景層」放在最下層獨立微調細節，再用「星點遮罩」把星點單獨控選出來疊回，就能完美掌握星雲與星點各自的亮度比例！

### ---

**📄 授權條款**

This project is open-source and available under the [MIT License](http://docs.google.com/LICENSE). Contributions, forks, and pull requests are highly welcome\!
