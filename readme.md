# 利用演化多工增強式學習解決模擬控制機器人問題
### Evolutionary Multitask Reinforcement Learning for Simulated Robot Control Problem

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

這是輔仁大學資訊工程學系111學年度的畢業專題，旨在探索如何利用演化多工（Evolutionary Multitasking）來增強深度強化學習（Deep Reinforcement Learning, DRL）在模擬機器人控制問題上的效能與效率。

**指導教授**: 廖容佐 博士
**組員**: 羅鈺婷、葉承翰、楊謹芳、劉懷萱

---

## 專題目標 (Project Goal)

近年來，仿生機器人的應用日益廣泛，從工業負載到救災活動都能看到其身影。然而，訓練這些機器人需要耗費大量的時間與運算資源，尤其是在獎勵稀疏（Sparse Rewards）的環境中，傳統的深度強化學習（DRL）方法經常面臨學習效率低落的挑戰。

本專案的核心目標是：**透過引入「演化多工」的概念，讓多個機器人控制任務在訓練過程中共享知識，藉此加速學習過程並提升最終效能**。

## 我們的解決方案 (Our Approach)

為了解決上述挑戰，我們提出了一種結合「演化式強化學習」與「演化多工」的混合演算法。

### 1. 核心框架：演化式強化學習 (ERL & CEM-RL)

我們以兩種先進的演化式強化學習（Evolutionary Reinforcement Learning, ERL）演算法作為基礎：

- **ERL (Evolution-Guided Policy Gradient)**：此方法利用演化演算法（EA）維護一個策略族群（Population），透過突變和交配產生多樣化的經驗來訓練一個 DRL 智能體（Agent）。同時，這個 DRL 智能體的梯度資訊也會被定期注入回族群中，結合了 EA 的廣域探索能力和 DRL 的高效優化能力。
- **CEM-RL (Cross-Entropy Method RL)**：此方法採用交叉熵方法（CEM）來指導策略族群的演化，每次迭代時，它會選出表現最好的「菁英」個體，並基於這些菁英的參數分佈來生成下一代，從而穩定地朝著更優的策略進行搜索。

### 2. 創新之處：演化多工 (SBO)

在本專案中，我們創新地將 **SBO (Symbiosis in Biocoenosis Optimization)** 演算法融入上述框架。SBO 是一種演化多工演算法，其核心思想源自於生物學中的「共生關係」。

當我們同時訓練兩個不同的任務時（例如 `Ant-v2` 和 `HalfCheetah-v2`），SBO 會：
1.  在每一代演化後，評估一個任務中的個體（策略）在另一個任務中的表現好壞（有益、有害或中性）。
2.  基於這些跨任務的表現，建立起兩個任務族群間的「共生關係」（如互利共生、片利共生、競爭等）。
3.  最後，根據這些關係動態計算出一個「轉移率 (Transfer Rate)」，決定要從表現好的任務中轉移多少「知識」（也就是菁英個體）給另一個任務，從而實現跨任務學習。

最終，我們提出了 **ERL-SBO** 與 **CEM-RL-SBO** 兩種新演算法，其實作流程可參考[專題報告](A06專題報告最終版.pdf)中的圖8與圖9。

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/3b783f8b-de10-4b12-aec7-9fd223eaffd8" alt="ERL-SBO" />
      <br />
      <sub><b>ERL-SBO</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/25a547c2-215a-4c11-9fe3-f924eb0ae2a8" alt="CEM-RL-SBO" />
      <br />
      <sub><b>CEM-RL-SBO</b></sub>
    </td>
  </tr>
</table>

## 實驗結果與展示 (Results & Demo)

我們在四個 MuJoCo 模擬環境中進行了六種兩兩組合的演化多工實驗。實驗結果表明，引入 SBO 確實能在多數情況下帶來效能提升。

**主要發現**：**CEM-TD3-SBO** 是整體表現最佳的演算法。相較於未加入多工前的 CEM-TD3，**在 12 組測試任務中有 9 組獲得了更高的最終平均獎勵**。

以下為 `Ant-v2` 與 `HalfCheetah-v2` 的訓練過程展示與學習曲線圖，訓練前的模型無法前進，在訓練後則能夠順利前進：

<table>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/97a6b539-a106-479c-8f90-1e9f3dedd137" alt="Ant-v2 模型展示" />
      <br />
      <sub><b>Ant-v2 模型訓練前</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/745fd300-b940-42fe-8b27-515edc443387" alt="Ant-v2 模型展示" />
      <br />
      <sub><b>Ant-v2 模型訓練後</b></sub>
    </td>
  </tr>
    <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/c3d8bb0e-fbdd-4bc9-8b04-5d608c18b56c" alt="HalfCheetah-v2 模型展示" />
      <br />
      <sub><b>HalfCheetah-v2 模型訓練前</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/495743b2-582d-4ebc-a49c-3087caeb3951" alt="HalfCheetah-v2 模型展示" />
      <br />
      <sub><b>HalfCheetah-v2 模型訓練後</b></sub>
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/19ffbdca-b0a4-46b3-b440-0272400df6b0" alt="Ant-v2 學習曲線" />
      <br />
      <sub><b>Ant-v2 學習曲線</b></sub>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/129b1925-04e2-4fa3-805b-261037dcd329" alt="HalfCheetah-v2 學習曲線" />
      <br />
      <sub><b>HalfCheetah-v2 學習曲線</b></sub>
    </td>
  </tr>
</table>

更詳細的數據與討論，請參考儲存庫中的[**專題報告**](A06專題報告最終版.pdf)。

## 未來展望 (Future Work)
根據我們的實驗與觀察，未來有幾個值得繼續研究的方向：

擴展至更多任務：受限於硬體，目前僅進行兩個任務的多工訓練。未來可嘗試同時處理更多任務，觀察知識轉移的複雜動態。

優化跨維度個體映射：目前在交換不同維度（state/action space）的個體時，採用的是直接對齊的方式。未來可以研究更智能的映射（Mapping）方法，以提升轉移學習的效果。

嘗試其他演算法：可以嘗試結合其他的演化多工演算法，或是將 SBO 應用於 PPO、TRPO 等其他 DRL 演算法上。

## 參考資料 (References)
[1] Khadka S. and Tumer K., "Evolution-guided policy gradient in reinforcement learning," in NeurIPS, 2018.
[2] Pourchot A. and Sigaud O., "CEM-RL: Combining evolutionary and gradient -based methods for policy search," in ICLR, 2019.
[3] R.-T. Liaw and C.-K. Ting, "Evolutionary manytasking optimization based on symbiosis in biocoenosis," in Thirty-Third AAAI Conference on Artificial Intelligence, 2019.

## License
This project is licensed under the MIT License.