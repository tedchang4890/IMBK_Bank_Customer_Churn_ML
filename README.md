# 고객 이탈 분류 ML 및 인사이트 분석

> 기간 : **2026.04.09.**

# 1. Tech Stack & Tools
### **Language & Environment**
- **Python** : 데이터 전처리 및 머신러닝 모델 구축의 메인 언어
### **Data Analysis & Manipulation**
- **Pandas & NumPy** : 대규모 은행 고객 데이터셋 핸들링 및 피처 엔지니어링
- **Matplotlib & Seaborn** : 데이터 분포 확인 및 모델 결과 해석을 위한 시각화
### **Machine Learning**
- **CatBoost / LightGBM / GradientBoostingClassifier / XGBoost** : 고성능 부스팅 알고리즘들을 전방 모델(Base Models)로 사용하여 예측력 강화
- **Scikit-learn** : StackingClassifier를 활용한 앙상블 모델 구현, LogisticRegression을 통한 메타 모델링, 성능 평가 지표(F1-score, Accuracy) 산출
- **Ensemble Technique** : 과적합 방지를 위해 전방 모델(Boosting 계열)과 후방 모델(Linear 계열)을 조합한 스태킹(Stacking) 전략 수립
  
# 2. Data Source
  * **데이터 출처**: 캐글 Bank Customer Churn Dataset (row: 10000, col:12)

# 3. Data Processing

### Feature Selection & Data Cleaning

  * **불필요한 식별자 제거** : 예측에 유의미한 정보를 제공하지 않는 **RowNumber, CustomerId, Surname** 컬럼을 삭제하여 모델의 복잡도를 줄이고 과적합을 방지
  * **결측치 및 이상치 확인** : 데이터의 무결성을 점검하여 분석의 신뢰도를 확보

### Encoding

  * **Label Encoding** : Gender(Female: 0, Male: 1)와 country(France: 0, Germany: 1, Spain: 2)로 변수들을 수치화하여 구성

### Feature Scaling

  * **Scaling** : StandardScaler를 통해 이상치의 영향을 감소

# 4. EDA

<img width="865" height="769" alt="스크린샷 2026-04-10 13 09 34" src="https://github.com/user-attachments/assets/bac3e152-3242-4543-bd14-9372cee3c6f7" />

변수들의 히트맵을 통해 타겟 변수로 설정한 **churn**과 상관관계 상위 4개인 **age, active_member, balance, gender**를 확인

<img width="882" height="754" alt="스크린샷 2026-04-10 13 10 09" src="https://github.com/user-attachments/assets/d065efef-0560-4933-b4c6-8836a55b59f1" />

- **age** : 유지고객의 경우 연령대 그래프의 첨도가 왼쪽에 위치하고 있고 이탈고객의 경우 첨도가 오른쪽에 있다. 따라서 유지고객은 연령대가 상대적으로 낮고 이탈고객은 연령대가 상대적으로 높아 이탈고객일수록 연령대가 높아진다는 것을 알 수 있다. -> **중장년층의 이탈고객 다수 분포하는 모습은 중장년층의 높은 투자선호도로 인해 은행을 이탈해 증권으로 옮길 확률이 높다고 예상된다.**

- **active_member** : 비활성 고객은 활성 고객 대비 유지 고객의 비율이 상대적으로 높고 이탈 고객의 비율이 상대적으로 낮다. 따라서 이탈고객 중 활성 고객이 더 많다는 것을 알 수 있다.-> **비활성 고객의 경우 휴면으로 관리가 되어 이탈되지 않아 표시 상 활성 고객의 이탈이 더 많아 보인다고 예상된다.**

- **balance** : 자산이 0에 근접한 경우 이탈고객과 유지고객이 가장 많고 이탈고객과 유지고객 모두 자산이 어느정도 이상이 되면 종모양의 그래프를 그린다. 따라서 이탈고객과 유지고객의 자산 분포는 유사하다고 볼 수 있다. -> **0에 많이 분포하는 그룹은 해당 은행이 주거래 은행이 아닌 고객이 많을 것으로 예상되고 오른쪽 종모양의 그래프의 경우 자산이 동일하다는 가정 하 이탈고객과 유지고객이 특정 비율을 유지하고 있어서 해당 종모양 그래프가 나온다고 예상된다.**

- **gender** : 여성의 경우 남성 대비 유지 고객의 비율이 상대적으로 낮고 이탈 고객의 비율이 상대적으로 높다. 따라서 이탈고객 중 여성이 더 많다는 것을 알 수 있다. -> **여성의 경우 조금 더 정보에 민감하여 조금 더 높은 금리 상품 등 이탈요인을 남성보다 더 많이 접할 가능성이 커서 여성고객의 이탈이 더 많아 보인다고 예산된다.**

# 5. ML

## AutoML 

- **Target Objective** : pycaret을 통한 F1-Score값

- **Search Space** : n_estimators, max_depth, learning_rate 등 각 부스팅 알고리즘의 핵심 파라미터 최적화

- **Target Models** : Target Objective 상위 CatBoost, LightGBM, GradientBoosting, XGBoost








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
