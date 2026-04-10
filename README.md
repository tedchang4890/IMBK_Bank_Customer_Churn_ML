# 고객 이탈 분류 ML 및 인사이트 분석

> **2026.04.09.**

# 🛠 Tech Stack & Tools
## **Language & Environment**
- Python : 데이터 전처리 및 머신러닝 모델 구축의 메인 언어
## **Data Analysis & Manipulation**
- Pandas & NumPy
- Matplotlib & Seaborn 
## **Machine Learning**
- **XGBoost / LightGBM / CatBoost** : 고성능 부스팅 알고리즘들을 전방 모델(Base Models)로 사용하여 예측력 강화
- Scikit-learn : StackingClassifier를 활용한 앙상블 모델 구현, LogisticRegression을 통한 메타 모델링, 성능 평가 지표(F1-score, Accuracy) 산출
- Ensemble Technique : 과적합 방지를 위해 전방 모델(Boosting 계열)과 후방 모델(Linear 계열)을 조합한 스태킹(Stacking) 전략 수립
  
# 📌 Project Overview

  * **데이터 출처**: [Kaggle Bank Customer Churn Dataset](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)
  * **핵심 과제**: 고객의 신용 점수, 거주 국가, 성별, 보유 자산 등 10개 이상의 특성(Feature)을 바탕으로 이탈 여부(Exited)를 분류

## 🔍 Analysis Process

### 1\. Data Preprocessing & Feature Engineering

데이터의 가독성과 모델의 학습 효율을 위해 다음과 같은 전처리를 수행했습니다.

  * **범주형 변수 변환**: 성별(Female: 0, Male: 1) 및 멤버십 활성화 여부(Active: 0, Inactive: 1) 등 이진 분류가 가능하도록 인코딩을 진행했습니다.
  * **불필요한 컬럼 제거**: 예측에 유의미한 영향을 주지 않는 고객 ID, 성(Surname) 등의 필드를 삭제하여 모델의 일반화 성능을 높였습니다.

### 2\. Modeling Strategy: Stacking Ensemble

단일 모델의 한계를 극복하기 위해 다층 구조의 **Stacking Classifier**를 설계했습니다.

  * **Base Models (1단계)**: `CatBoost`, `LightGBM`, `GradientBoosting`, `XGBoost`를 활용하여 데이터의 다양한 패턴을 개별적으로 학습했습니다.
  * **Meta Model (2단계)**: 개별 모델들의 예측 결과를 결합할 때 발생할 수 있는 과적합(Overfitting)을 방지하기 위해, 최종 모델로 **Logistic Regression**을 사용했습니다.

## 📈 Evaluation Results

최종 스태킹 모델을 검증 데이터에 적용한 결과는 다음과 같습니다.

  * **Accuracy Score**: `0.8715` (약 87%의 예측 정확도 달성)
  * **F1-Score**: `0.6112`

정확도 측면에서는 매우 우수한 성능을 보였으나, 이탈 고객에 대한 정밀한 탐지 능력을 의미하는 F1-Score는 상대적으로 낮게 나타났습니다. 이는 클래스 불균형(이탈 고객 수의 부족) 문제로 판단되며, 향후 오버샘플링(SMOTE) 기술 적용을 통해 개선할 여지를 확인했습니다.

## 💡 Key Insights

  * 고객의 활동성(Active Member) 여부가 이탈 예측의 중요한 변수로 작용함을 확인했습니다.
  * 강력한 부스팅 모델들을 조합한 스태킹 기법이 단일 모델 대비 안정적인 성능을 제공함을 증명했습니다.

-----

### 📂 Repository Structure

  * `머신러닝_컴페티션_BaseLine.ipynb`: 데이터 분석 및 모델링 전체 코드가 포함된 주피터 노트북
  * `data/`: 분석에 사용된 원본 데이터셋 (Kaggle 제공)
