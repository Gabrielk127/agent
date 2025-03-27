import google.generativeai as genai
import json
from GEMINI_KEY import GEMINI_API_KEY

# Configura a API key
API_KEY = GEMINI_API_KEY
genai.configure(api_key=API_KEY)

# Define o modelo do Gemini
model = genai.GenerativeModel("gemini-2.0-flash")

# Carrega histórico de conversa com tratamento de erros
def load_history():
    try:
        with open("chat_history.json", "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

# Salva o histórico atualizado
def save_history(history):
    with open("chat_history.json", "w") as file:
        json.dump(history, file, indent=4)

# Carrega treinamento salvo com tratamento de erros
def load_training():
    try:
        with open("training_data.json", "r") as file:
            return json.load(file).get("content", "")
    except (FileNotFoundError, json.JSONDecodeError):
        return ""

# Salva treinamento em arquivo separado
def save_training(training_text):
    with open("training_data.json", "w") as file:
        json.dump({"content": training_text}, file, indent=4)

# Adiciona novo contexto ao existente
def add_training(new_context):
    existing_context = load_training()
    updated_context = existing_context + "\n" + new_context if existing_context else new_context
    save_training(updated_context)
    return updated_context

# Converte o histórico para o formato aceito pelo Gemini
# O contexto agora é parte de cada mensagem do usuário
def format_history_for_gemini(history, training_context):
    formatted_history = []
    for msg in history:
        if msg["role"] == "user" and training_context:
            msg["content"] = f"Contexto: {training_context}\n{msg['content']}"
        formatted_history.append({"role": msg["role"], "parts": [{"text": msg["content"]}]})
    return formatted_history

# Inicializa histórico e carrega treinamento
chat_history = load_history()
training_context = load_training()

print("🤖 Chat Gemini (digite 'sair' para encerrar, 'treinar' para ensinar o bot, 'adicionar treino' para expandir o treinamento, 'resetar treino' para apagar o treinamento)")

while True:
    user_message = input("Você: ")

    if user_message.lower() == "sair":
        print("Encerrando conversa. Histórico salvo!")
        save_history(chat_history)
        break

    # MODO TREINAMENTO
    if user_message.lower() == "treinar":
        print("🧠 Modo treinamento ativado! Digite as informações para o bot aprender (digite 'fim' para terminar).")
        training_data = []
        while True:
            info = input("Treine o bot: ")
            if info.lower() == "fim":
                break
            training_data.append(info)

        # Salva o novo treinamento
        training_context = " ".join(training_data)
        save_training(training_context)
        print("✅ Treinamento salvo! Agora o bot usará esse contexto.")
        continue

    # Adicionar mais contexto ao treinamento
    if user_message.lower() == "adicionar treino":
        print("➕ Adicionar mais contexto ao treinamento. Digite o novo conteúdo (digite 'fim' para terminar):")
        new_training_data = []
        while True:
            info = input("Novo treinamento: ")
            if info.lower() == "fim":
                break
            new_training_data.append(info)

        # Adiciona o novo contexto ao existente
        training_context = add_training(" ".join(new_training_data))
        print("✅ Novo contexto adicionado ao treinamento!")
        continue

    # Resetar o treinamento
    if user_message.lower() == "resetar treino":
        save_training("")
        training_context = ""
        print("🔄 Treinamento resetado. O bot agora não tem contexto especializado.")
        continue

    # Adiciona a mensagem do usuário ao histórico
    chat_history.append({"role": "user", "content": user_message})

    # Prepara o histórico no formato correto para o Gemini
    formatted_history = format_history_for_gemini(chat_history, training_context)

    # Gera a resposta usando o histórico formatado
    try:
        response = model.generate_content(formatted_history)
        bot_reply = response.text
    except Exception as e:
        bot_reply = "Desculpe, houve um erro ao processar a resposta. Tente novamente."
        print(f"Erro: {e}")

    print(f"Gemini: {bot_reply}\n")

    # Salva a resposta no histórico
    chat_history.append({"role": "assistant", "content": bot_reply})

    # Atualiza o histórico no arquivo
    save_history(chat_history)

# Adiciona o novo formato de análise Business Model Canvas ao treinamento
canvas_training_data = "O chatbot agora é um agente especializado na análise de Business Model Canvas. Ele identifica falhas conceituais, aponta inconsistências e sugere melhorias estratégicas. A análise é organizada por tópicos: Proposta de Valor, Segmentos de Mercado, Canais, Relação com o Cliente, Fontes de Renda, Recursos-chave, Atividades-chave, Parceiros-chave, Estrutura de Custos e Conclusão e Recomendações. As respostas são objetivas, estruturadas e fornecem exemplos práticos, quando aplicável. Além disso, ele gera relatórios em PDF com cabeçalho e rodapé personalizados."
add_training(canvas_training_data)