# 고객 이탈 분류 ML 및 인사이트 분석

> **2026.04.09.**

# 1. Tech Stack & Tools
### **Language & Environment**
- **Python** : 데이터 전처리 및 머신러닝 모델 구축의 메인 언어
### **Data Analysis & Manipulation**
- **Pandas & NumPy** : 대규모 은행 고객 데이터셋 핸들링 및 피처 엔지니어링
- **Matplotlib & Seaborn** : 데이터 분포 확인 및 모델 결과 해석을 위한 시각화
### **Machine Learning**
- **XGBoost / LightGBM / CatBoost** : 고성능 부스팅 알고리즘들을 전방 모델(Base Models)로 사용하여 예측력 강화
- **Scikit-learn** : StackingClassifier를 활용한 앙상블 모델 구현, LogisticRegression을 통한 메타 모델링, 성능 평가 지표(F1-score, Accuracy) 산출
- **Ensemble Technique** : 과적합 방지를 위해 전방 모델(Boosting 계열)과 후방 모델(Linear 계열)을 조합한 스태킹(Stacking) 전략 수립
  
# 2. Data Source
  * **데이터 출처**: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

# 3. Data Ptocessing

### Feature Selection & Data Cleaning

  * **불필요한 식별자 제거** : 예측에 유의미한 정보를 제공하지 않는 **RowNumber, CustomerId, Surname** 컬럼을 삭제하여 모델의 복잡도를 줄이고 과적합을 방지
  * **결측치 및 이상치 확인** : 데이터의 무결성을 점검하여 분석의 신뢰도를 확보

### Encoding

  * **Label Encoding** : Gender(Female: 0, Male: 1)와 country(France: 0, Germany: 1, Spain: 2)로 변수들을 수치화하여 구성

### Feature Scaling

  * **Scaling** : StandardScaler를 통해 이상치의 영향을 감소

## 3. EDA

<img width="865" height="769" alt="스크린샷 2026-04-10 13 09 34" src="https://github.com/user-attachments/assets/bac3e152-3242-4543-bd14-9372cee3c6f7" />

변수들의 히트맵을 통해 타겟 변수로 설정한 **churn**과 상관관계 상위 4개인 age, active_member, balance, gender를 확인

<img width="882" height="754" alt="스크린샷 2026-04-10 13 10 09" src="https://github.com/user-attachments/assets/d065efef-0560-4933-b4c6-8836a55b59f1" />



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
