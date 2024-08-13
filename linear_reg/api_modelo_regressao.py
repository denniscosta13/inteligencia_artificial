from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instancia do FastAPI
app = FastAPI()

# Criar uma classe com os dados da request
class request_body(BaseModel):
    horas_estudo: float

# Carregar o modelo pra realizar predict
modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

@app.post("/predict")
def predict(data : request_body):
    # Preparar os dados para predição

    input_feature = [[data.horas_estudo]]

    #Realizar a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

    return {
        'pontuacao_teste': y_pred.tolist()
    }
