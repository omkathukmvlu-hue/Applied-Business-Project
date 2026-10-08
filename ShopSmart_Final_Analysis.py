# ============================================================
# SHOPSMART - FINAL PYTHON ANALYSIS
# E-Commerce Sales & Customer Analytics
# Dataset: ShopSmart_Ecommerce_Sales_Dataset.csv (8000 rows)
# This code uses the EXACT fields in the ShopSmart Tableau project.
# ============================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import milp, LinearConstraint, Bounds
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm
from statsmodels.formula.api import ols

warnings.filterwarnings("ignore")
np.random.seed(45)

DATASET_FILE = "ShopSmart_Ecommerce_Sales_Dataset.csv"
OUTPUT_DIR = "ShopSmart_Python_Outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------
df = pd.read_csv(DATASET_FILE)
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

numeric_cols = [
    "Customer_Age", "Quantity", "Unit_Price", "Discount_Percent",
    "Sales", "Profit", "Customer_Rating", "Delivery_Days",
    "Inventory_Units"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

print("\n" + "="*70)
print("SHOPSMART E-COMMERCE SALES & CUSTOMER ANALYTICS")
print("="*70)
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing values:", int(df.isna().sum().sum()))
print("Duplicate rows:", int(df.duplicated().sum()))

# ------------------------------------------------------------
# 2. DATA CLEANING & FEATURE ENGINEERING
# ------------------------------------------------------------
df = df.drop_duplicates().copy()
df = df.dropna(subset=["Order_Date", "Sales", "Profit"])
df["Year"] = df["Order_Date"].dt.year
df["Month_Number"] = df["Order_Date"].dt.month
df["Year_Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Profit_Margin"] = np.where(df["Sales"] != 0, df["Profit"] / df["Sales"] * 100, 0)
df["Returned_Flag"] = (df["Return_Status"] == "Returned").astype(int)
df["Promotion_Flag"] = df["Promotion"].fillna("").ne("").astype(int)

# ------------------------------------------------------------
# 3. BUSINESS KPIs
# ------------------------------------------------------------
total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
total_orders = df["Order_ID"].nunique()
total_customers = df["Customer_ID"].nunique()
aov = total_sales / total_orders
avg_rating = df["Customer_Rating"].mean()
return_rate = df["Returned_Flag"].mean() * 100
avg_delivery = df["Delivery_Days"].mean()

print("\nBUSINESS KPIs")
print(f"Total Sales        : ₹{total_sales:,.2f}")
print(f"Total Profit       : ₹{total_profit:,.2f}")
print(f"Total Orders       : {total_orders:,}")
print(f"Total Customers    : {total_customers:,}")
print(f"Average Order Value: ₹{aov:,.2f}")
print(f"Average Rating     : {avg_rating:.2f}")
print(f"Return Rate        : {return_rate:.2f}%")
print(f"Average Delivery   : {avg_delivery:.2f} days")

# ------------------------------------------------------------
# 4. EDA / TABLEAU-SUPPORTING ANALYSIS
# ------------------------------------------------------------
category_sales = df.groupby("Product_Category")["Sales"].sum().sort_values(ascending=False)
customer_sales = df.groupby("Customer_Type")["Sales"].sum().sort_values(ascending=False)
promotion_sales = df.groupby(df["Promotion"].replace("", "No Promotion"))["Sales"].sum().sort_values(ascending=False)
return_sales = df.groupby("Return_Status")["Sales"].sum().sort_values(ascending=False)
city_sales = df.groupby("City")["Sales"].sum().sort_values(ascending=False)
product_sales = df.groupby("Product_Name")["Sales"].sum().sort_values(ascending=False).head(10)

print("\nSales by Product Category:\n", category_sales)
print("\nSales by Customer Type:\n", customer_sales)
print("\nSales by Promotion:\n", promotion_sales)
print("\nSales by Return Status:\n", return_sales)
print("\nTop 10 Products:\n", product_sales)

# ------------------------------------------------------------
# 5. VISUALIZATIONS
# ------------------------------------------------------------
plt.figure(figsize=(9,5))
category_sales.plot(kind="bar")
plt.title("Sales by Product Category")
plt.xlabel("Product Category"); plt.ylabel("Sales"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_sales_by_category.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
customer_sales.plot(kind="bar")
plt.title("Customer Type Analysis")
plt.xlabel("Customer Type"); plt.ylabel("Sales"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_customer_type_sales.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
plt.scatter(df["Customer_Age"], df["Sales"], alpha=0.25)
plt.title("Customer Age vs Sales")
plt.xlabel("Customer Age"); plt.ylabel("Sales"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_age_vs_sales.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
promotion_sales.plot(kind="barh")
plt.title("Promotion Performance")
plt.xlabel("Sales"); plt.ylabel("Promotion"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_promotion_performance.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
plt.scatter(df["Discount_Percent"], df["Sales"], alpha=0.25)
plt.title("Discount vs Sales")
plt.xlabel("Discount Percentage"); plt.ylabel("Sales"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_discount_vs_sales.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
plt.hist(df["Customer_Rating"].dropna(), bins=10)
plt.title("Customer Ratings Distribution")
plt.xlabel("Customer Rating"); plt.ylabel("Frequency"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_customer_ratings.png", dpi=180); plt.show(); plt.close()

plt.figure(figsize=(9,5))
plt.hist(df["Delivery_Days"].dropna(), bins=10)
plt.title("Delivery Performance")
plt.xlabel("Delivery Days"); plt.ylabel("Number of Orders"); plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_delivery_performance.png", dpi=180); plt.show(); plt.close()

# ------------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------
stats_cols = ["Customer_Age", "Quantity", "Unit_Price", "Discount_Percent", "Sales", "Profit", "Customer_Rating", "Delivery_Days", "Inventory_Units"]
desc = df[stats_cols].describe().T
desc["Median"] = df[stats_cols].median()
desc["Variance"] = df[stats_cols].var()
desc["Skewness"] = df[stats_cols].skew()
print("\nDESCRIPTIVE STATISTICS\n", desc)
desc.to_csv(f"{OUTPUT_DIR}/descriptive_statistics.csv")

# ------------------------------------------------------------
# 7. CORRELATION
# ------------------------------------------------------------
corr_cols = ["Quantity", "Unit_Price", "Discount_Percent", "Sales", "Profit", "Customer_Rating", "Delivery_Days", "Inventory_Units"]
corr = df[corr_cols].corr()
print("\nCORRELATION MATRIX\n", corr)
corr.to_csv(f"{OUTPUT_DIR}/correlation_matrix.csv")
plt.figure(figsize=(10,7))
plt.imshow(corr, aspect="auto")
plt.colorbar(label="Correlation")
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.index)), corr.index)
for i in range(len(corr)):
    for j in range(len(corr)):
        plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("ShopSmart Correlation Matrix")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/08_correlation_matrix.png", dpi=180); plt.show(); plt.close()

# ------------------------------------------------------------
# 8. MULTIPLE LINEAR REGRESSION / ML
# ------------------------------------------------------------
features = ["Customer_Age", "Quantity", "Unit_Price", "Discount_Percent", "Delivery_Days"]
ml = df[features + ["Sales"]].dropna()
X = ml[features]; y = ml["Sales"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=45)
model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)
r2 = r2_score(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
mae = mean_absolute_error(y_test, pred)
print("\nLINEAR REGRESSION")
print(f"R²   : {r2:.4f}")
print(f"RMSE : ₹{rmse:,.2f}")
print(f"MAE  : ₹{mae:,.2f}")
for f, c in zip(features, model.coef_): print(f"{f}: {c:.4f}")
plt.figure(figsize=(8,6)); plt.scatter(y_test,pred,alpha=.25)
mn=min(y_test.min(),pred.min()); mx=max(y_test.max(),pred.max()); plt.plot([mn,mx],[mn,mx])
plt.title(f"Actual vs Predicted Sales (R²={r2:.4f})"); plt.xlabel("Actual Sales"); plt.ylabel("Predicted Sales")
plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/09_regression_actual_vs_predicted.png", dpi=180); plt.show(); plt.close()

# OLS summary
ols_df = df[["Sales", "Quantity", "Unit_Price", "Discount_Percent"]].dropna()
ols_model = ols("Sales ~ Quantity + Unit_Price + Discount_Percent", data=ols_df).fit()
print("\nOLS REGRESSION\n", ols_model.summary())
with open(f"{OUTPUT_DIR}/ols_summary.txt", "w", encoding="utf-8") as f: f.write(str(ols_model.summary()))

# ------------------------------------------------------------
# 9. PROBABILITY ANALYSIS
# ------------------------------------------------------------
q75 = df["Sales"].quantile(.75)
prob_high_sales = (df["Sales"] > q75).mean()
prob_return = (df["Return_Status"] == "Returned").mean()
prob_returning = (df["Customer_Type"] == "Returning").mean()
prob_weekend = (df["Day_Type"] == "Weekend").mean()
print("\nPROBABILITY")
print(f"P(Sales > 75th percentile): {prob_high_sales:.4f}")
print(f"P(Returned): {prob_return:.4f}")
print(f"P(Returning Customer): {prob_returning:.4f}")
print(f"P(Weekend Order): {prob_weekend:.4f}")

# ------------------------------------------------------------
# 10. DECISION MAKING UNDER UNCERTAINTY / EMV
# ------------------------------------------------------------
decision = pd.DataFrame({
    "Alternative": ["Premium Promotion", "Standard Promotion"],
    "High_Demand_Probability": [0.40, 0.40],
    "High_Demand_Payoff": [600000, 350000],
    "Low_Demand_Probability": [0.60, 0.60],
    "Low_Demand_Payoff": [-120000, 80000]
})
decision["EMV"] = decision["High_Demand_Probability"]*decision["High_Demand_Payoff"] + decision["Low_Demand_Probability"]*decision["Low_Demand_Payoff"]
print("\nEMV DECISION\n", decision)
print("Recommended decision:", decision.loc[decision.EMV.idxmax(), "Alternative"])

# Decision tree image
fig, ax = plt.subplots(figsize=(12,7)); ax.axis("off")
root=(.08,.5); p=(.42,.7); s=(.42,.3); ph=(.72,.86); pl=(.72,.58); sh=(.72,.42); sl=(.72,.14)
for a,b in [(root,p),(root,s),(p,ph),(p,pl),(s,sh),(s,sl)]: ax.annotate("",xy=b,xytext=a,arrowprops=dict(arrowstyle="->",linewidth=1.8))
ax.text(*root,"DECISION\nChoose Promotion Strategy",ha="center",va="center",bbox=dict(boxstyle="round",fill=False),fontsize=11,weight="bold")
ax.text(*p,"Premium Promotion\nEMV = ₹168,000",ha="center",va="center",bbox=dict(boxstyle="round",fill=False),fontsize=10)
ax.text(*s,"Standard Promotion\nEMV = ₹188,000",ha="center",va="center",bbox=dict(boxstyle="round",fill=False),fontsize=10)
for pos,txt in [(ph,"High Demand\nP=0.40\nPayoff=₹600,000"),(pl,"Low Demand\nP=0.60\nPayoff=−₹120,000"),(sh,"High Demand\nP=0.40\nPayoff=₹350,000"),(sl,"Low Demand\nP=0.60\nPayoff=₹80,000")]: ax.text(*pos,txt,ha="center",va="center",bbox=dict(boxstyle="round",fill=False),fontsize=9)
ax.set_title("Decision Tree — Expected Monetary Value",fontsize=15,weight="bold")
ax.text(.5,.03,"Recommendation: Maintain Standard Promotion (EMV ₹188,000)",ha="center",weight="bold")
plt.savefig(f"{OUTPUT_DIR}/10_decision_tree_EMV.png",dpi=220,bbox_inches="tight"); plt.show(); plt.close()

# ------------------------------------------------------------
# 11. SAMPLING + CENTRAL LIMIT THEOREM
# ------------------------------------------------------------
sample_size=50; repetitions=1000
sample_means=np.array([df["Sales"].sample(sample_size,replace=False,random_state=45+i).mean() for i in range(repetitions)])
print("\nCLT")
print(f"Population mean: ₹{df.Sales.mean():,.2f}")
print(f"Mean of sample means: ₹{sample_means.mean():,.2f}")
print(f"Theoretical SE: ₹{df.Sales.std(ddof=1)/np.sqrt(sample_size):,.2f}")
plt.figure(figsize=(9,5)); plt.hist(sample_means,bins=30); plt.axvline(df.Sales.mean(),ls="--",label="Population Mean"); plt.title("Sampling Distribution of Mean Sales"); plt.xlabel("Sample Mean"); plt.ylabel("Frequency"); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/11_clt_sampling_distribution.png",dpi=180); plt.show(); plt.close()

# ------------------------------------------------------------
# 12. CONFIDENCE INTERVALS
# ------------------------------------------------------------
sales=df.Sales.dropna(); n=len(sales); mean=sales.mean(); se=stats.sem(sales)
print("\nCONFIDENCE INTERVALS")
for level in [.90,.95,.99]: print(f"{int(level*100)}% CI:", stats.t.interval(level,df=n-1,loc=mean,scale=se))

# ------------------------------------------------------------
# 13. BOOTSTRAPPING
# ------------------------------------------------------------
boot=np.array([np.median(np.random.choice(sales.values,n,replace=True)) for _ in range(1000)])
boot_ci=np.percentile(boot,[2.5,97.5])
print("\nBOOTSTRAP")
print("Observed median:", sales.median()); print("95% bootstrap CI:", boot_ci)
plt.figure(figsize=(9,5)); plt.hist(boot,bins=30); plt.axvline(boot_ci[0],ls="--",label="2.5%"); plt.axvline(boot_ci[1],ls="--",label="97.5%"); plt.title("Bootstrap Distribution of Median Sales"); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/12_bootstrap.png",dpi=180); plt.show(); plt.close()

# ------------------------------------------------------------
# 14. HYPOTHESIS TESTING
# H0: Mean Sales = ₹15,000
# ------------------------------------------------------------
t_stat,p_value=stats.ttest_1samp(sales,15000)
print("\nHYPOTHESIS TEST")
print(f"t-statistic: {t_stat:.4f}"); print(f"p-value: {p_value:.6g}"); print("Decision:","Reject H0" if p_value<.05 else "Fail to reject H0")

# ------------------------------------------------------------
# 15. A/B TESTING — PROMOTION VS NO PROMOTION
# ------------------------------------------------------------
promo=df[df.Promotion.fillna("").ne("")].Sales.dropna()
no_promo=df[df.Promotion.fillna("").eq("")].Sales.dropna()
t_ab,p_ab=stats.ttest_ind(promo,no_promo,equal_var=False)
print("\nA/B TEST")
print(f"Promotion mean sales: ₹{promo.mean():,.2f}")
print(f"No-promotion mean sales: ₹{no_promo.mean():,.2f}")
print(f"t-statistic: {t_ab:.4f}"); print(f"p-value: {p_ab:.6g}"); print("Decision:","Significant difference" if p_ab<.05 else "No significant difference")

# ------------------------------------------------------------
# 16. TIME SERIES + FORECAST ACCURACY
# ------------------------------------------------------------
monthly=df.set_index("Order_Date").resample("MS")["Sales"].sum().asfreq("MS")
train=monthly.iloc[:-4]; test=monthly.iloc[-4:]
holt=ExponentialSmoothing(train,trend="add",seasonal=None,initialization_method="estimated").fit()
holt_pred=holt.forecast(4)
arima=ARIMA(train,order=(1,1,1)).fit(); arima_pred=arima.forecast(4)

def metrics(actual,predicted):
    actual=np.asarray(actual); predicted=np.asarray(predicted)
    rmse=np.sqrt(np.mean((actual-predicted)**2)); mae=np.mean(np.abs(actual-predicted)); mape=np.mean(np.abs((actual-predicted)/actual))*100
    return rmse,mae,mape

hm=metrics(test,holt_pred); am=metrics(test,arima_pred)
print("\nFORECAST ACCURACY")
print(f"Holt RMSE ₹{hm[0]:,.2f}, MAE ₹{hm[1]:,.2f}, MAPE {hm[2]:.2f}%")
print(f"ARIMA RMSE ₹{am[0]:,.2f}, MAE ₹{am[1]:,.2f}, MAPE {am[2]:.2f}%")

holt_full=ExponentialSmoothing(monthly,trend="add",seasonal="add",seasonal_periods=12,initialization_method="estimated").fit()
future_holt=holt_full.forecast(6)
arima_full=ARIMA(monthly,order=(1,1,1)).fit(); future_arima=arima_full.forecast(6)
plt.figure(figsize=(12,6)); plt.plot(monthly.index,monthly.values,marker="o",label="Historical"); plt.plot(future_holt.index,future_holt.values,marker="o",ls="--",label="Holt-Winters"); plt.plot(future_arima.index,future_arima.values,marker="o",ls="--",label="ARIMA"); plt.title("Monthly Sales Forecast"); plt.xlabel("Month"); plt.ylabel("Sales"); plt.legend(); plt.xticks(rotation=45); plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/13_forecasting.png",dpi=180); plt.show(); plt.close()

# ------------------------------------------------------------
# 17. OPTIMIZATION — PRODUCT CATEGORY ALLOCATION
# ------------------------------------------------------------
cat=df.groupby("Product_Category").agg(Avg_Unit_Price=("Unit_Price","mean"),Avg_Profit_Per_Order=("Profit","mean"),Total_Units=("Quantity","sum"))
categories=cat.index.tolist(); profit_per_unit=cat.Avg_Profit_Per_Order.values; price=cat.Avg_Unit_Price.values; max_units=cat.Total_Units.values
budget=float((df.Unit_Price*df.Quantity).sum()*0.10)
objective=-profit_per_unit
constraints=[LinearConstraint(price.reshape(1,-1),-np.inf,budget),LinearConstraint(np.ones((1,len(categories))),max(1,int(df.Quantity.sum()*.05)),np.inf)]
result=milp(c=objective,integrality=np.ones(len(categories)),bounds=Bounds(np.zeros(len(categories)),max_units),constraints=constraints)
if result.success:
    allocation=np.rint(result.x).astype(int)
    opt=pd.DataFrame({"Product_Category":categories,"Optimal_Units":allocation,"Avg_Unit_Price":price,"Avg_Profit":profit_per_unit})
    opt["Estimated_Investment"]=opt.Optimal_Units*opt.Avg_Unit_Price
    opt["Estimated_Profit"]=opt.Optimal_Units*opt.Avg_Profit
    print("\nOPTIMIZATION\n",opt); print("Estimated investment:",opt.Estimated_Investment.sum()); print("Estimated profit:",opt.Estimated_Profit.sum())
    opt.to_csv(f"{OUTPUT_DIR}/optimization.csv",index=False)
else: print("Optimization failed:",result.message)

# ------------------------------------------------------------
# 18. MONTE CARLO SIMULATION
# ------------------------------------------------------------
np.random.seed(45); simulations=5000; months_ahead=12
mc_results=[]
for name,mult in {"Conservative":.90,"Expected":1.00,"High Demand":1.10}.items():
    values=np.maximum(np.random.normal(monthly.mean()*mult,monthly.std(),(simulations,months_ahead)),0).sum(axis=1)
    mc_results.append([name,values.mean(),np.median(values),np.percentile(values,5),np.percentile(values,95)])
mc=pd.DataFrame(mc_results,columns=["Scenario","Mean_12_Month_Revenue","Median","P5","P95"])
print("\nMONTE CARLO\n",mc); mc.to_csv(f"{OUTPUT_DIR}/monte_carlo.csv",index=False)

# ------------------------------------------------------------
# 19. ONE-WAY ANOVA
# ------------------------------------------------------------
groups=[g["Sales"].values for _,g in df.groupby("Product_Category")]
f_stat,p_anova=stats.f_oneway(*groups)
print("\nONE-WAY ANOVA")
print(f"F-statistic: {f_stat:.4f}"); print(f"p-value: {p_anova:.6g}"); print("Decision:","Reject H0" if p_anova<.05 else "Fail to reject H0")

# ------------------------------------------------------------
# 20. FACTORIAL EXPERIMENT / TWO-WAY ANOVA
# Factors: Customer Type and Promotion Status
# ------------------------------------------------------------
factor=df.copy(); factor["Promotion_Status"]=np.where(factor.Promotion.fillna("").eq(""),"No Promotion","Promotion")
factor_sample=factor.groupby(["Customer_Type","Promotion_Status"],group_keys=False).apply(lambda x:x.sample(min(len(x),500),random_state=45)).reset_index(drop=True)
model2=ols("Sales ~ C(Customer_Type) * C(Promotion_Status)",data=factor_sample).fit()
two_way=sm.stats.anova_lm(model2,typ=2)
print("\nTWO-WAY ANOVA / FACTORIAL EXPERIMENT\n",two_way)
cell=factor_sample.groupby(["Customer_Type","Promotion_Status"])["Sales"].mean().unstack()
plt.figure(figsize=(8,5))
for p in cell.columns: plt.plot(cell.index,cell[p],marker="o",label=p)
plt.title("Customer Type × Promotion Interaction"); plt.xlabel("Customer Type"); plt.ylabel("Mean Sales"); plt.legend(); plt.tight_layout(); plt.savefig(f"{OUTPUT_DIR}/14_factorial_experiment.png",dpi=180); plt.show(); plt.close()
two_way.to_csv(f"{OUTPUT_DIR}/two_way_anova.csv")

# ------------------------------------------------------------
# 21. CUSTOMER + DISCOUNT ANALYSIS
# ------------------------------------------------------------
print("\nCUSTOMER AGE")
print(f"Average age: {df.Customer_Age.mean():.2f}")
print(f"Median age : {df.Customer_Age.median():.2f}")
disc_corr=df[["Discount_Percent","Sales"]].corr().iloc[0,1]
print(f"Discount vs Sales correlation: {disc_corr:.4f}")

# ------------------------------------------------------------
# 22. SAVE CLEANED DATASET + SUMMARY
# ------------------------------------------------------------
df.to_csv(f"{OUTPUT_DIR}/ShopSmart_Cleaned_Dataset.csv",index=False)
summary={"Total Sales":total_sales,"Total Profit":total_profit,"Orders":total_orders,"Customers":total_customers,"AOV":aov,"Average Rating":avg_rating,"Return Rate %":return_rate,"Average Delivery Days":avg_delivery,"Regression R2":r2,"Regression RMSE":rmse,"Regression MAE":mae,"Hypothesis p-value":p_value,"AB p-value":p_ab,"ANOVA p-value":p_anova}
pd.Series(summary).to_csv(f"{OUTPUT_DIR}/project_summary.csv")

print("\n"+"="*70)
print("SHOPSMART ANALYSIS COMPLETED SUCCESSFULLY")
print("Outputs saved in:",OUTPUT_DIR)
print("="*70)
