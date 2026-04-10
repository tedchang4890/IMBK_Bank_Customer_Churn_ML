# 🏦 Bank Customer Churn Prediction

> **머신러닝 스태킹 앙상블 기법을 활용한 은행 고객 이탈 예측 프로젝트**

본 프로젝트는 고객의 금융 데이터와 인적 사항을 분석하여 이탈 여부를 예측하고, 이를 통해 은행의 고객 유지 전략(Retention Strategy) 수립에 기여하는 것을 목표로 합니다. 단순히 높은 정확도를 기록하는 것에 그치지 않고, 데이터 분석의 전체 프로세스 이해와 결과에 대한 설명력을 높이는 데 집중했습니다.

## 🛠 Tech Stack

## 📌 Project Overview

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
