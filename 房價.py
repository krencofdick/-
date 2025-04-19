import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import gradio as gr

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 選擇有用的欄位並處理
df = df[df["建物總面積"] > 0]
X = df[["土地面積", "建物總面積", "屋齡", "房數", "廳數", "衛數"]]
y = df["總價"]

# 建立模型
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 預測與特色建議函數
def predict_price(land_area, building_area, age, rooms, halls, bathrooms):
    input_data = [[land_area, building_area, age, rooms, halls, bathrooms]]
    predicted_price = model.predict(input_data)[0]
    
    # 特色建議（根據房間數或屋齡）
    if age < 5:
        feature = "新屋，裝潢機會多～✨"
    elif rooms >= 4:
        feature = "空間超大，適合大家庭 👨‍👩‍👧‍👦"
    else:
        feature = "價格親民，適合首購族 🏡"
    
    return f"預估總價：約 {int(predicted_price)} 萬元\n特色推薦：{feature}"

# 建立 Gradio 介面
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="土地面積 (坪)"),
        gr.Number(label="建物總面積 (坪)"),
        gr.Number(label="屋齡 (年)"),
        gr.Number(label="房數"),
        gr.Number(label="廳數"),
        gr.Number(label="衛數")
    ],
    outputs="text",
    title="🏠 台北市房價小幫手",
    description="輸入基本資訊，預測房價，並給你一點小建議！"
)

iface.launch()
