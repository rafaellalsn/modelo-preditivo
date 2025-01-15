from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Certifique-se de que o modelo .pkl (ou outro formato) esteja no mesmo diretório ou forneça o caminho correto
model = pickle.load(open('modelo_preditivo.pkl', 'rb'))  # Carrega o modelo preditivo

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        
        # Captura os dados do formulário
        idade = int(request.form.get("idade"))
        genero = request.form.get("genero")
        tempo_trabalho = int(request.form.get("tempo_trabalho"))
        setor = request.form.get("setor")

        # Pré-processamento dos dados, se necessário
        # Vamos assumir que o modelo recebe uma lista de recursos numéricos, talvez convertendo 'genero' e 'setor' para valores numéricos
        genero = 0 if genero == 'masculino' else 1  # Exemplo de codificação de gênero
        
        setor = 0 if setor == 'tecnologia' else 1 # Outros pré-processamentos poderiam ser necessários aqui

        # Passando os dados para o modelo preditivo
        features = np.array([[idade, genero, tempo_trabalho, setor]])  # Ajuste conforme o formato esperado pelo seu modelo
        previsao = model.predict(features)

        # Aqui podemos exibir a previsão ou a resposta
        return render_template("index.html", previsao=previsao[0])
    
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)