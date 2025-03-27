import os
import json
import markdown
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage
from langchain.memory import ConversationBufferMemory
from pdfminer.high_level import extract_text
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageTemplate, Frame, ListFlowable, ListItem
from bs4 import BeautifulSoup
from GEMINI_KEY import GEMINI_API_KEY

os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

def load_history():
    try:
        with open("chat_history.json", "r") as file:
            history_data = json.load(file)
            return [
                HumanMessage(content=msg["content"]) if msg["role"] == "user" 
                else AIMessage(content=msg["content"]) 
                for msg in history_data
            ]
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Erro ao carregar histórico: {e}")
        return []

def save_history(history):
    try:
        with open("chat_history.json", "w") as file:
            json.dump(
                [{"role": "user" if isinstance(msg, HumanMessage) else "assistant", "content": msg.content}
                 for msg in history],
                file,
                indent=4
            )
    except Exception as e:
        print(f"Erro ao salvar histórico: {e}")

def load_training_data(training_file="training_data.json"):
    try:
        with open(training_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        print("⚠️ Arquivo de treinamento não encontrado!")
        return {"content": "Agente sem treinamento prévio."}
    except json.JSONDecodeError:
        print("⚠️ Erro ao ler o arquivo JSON! Verifique a formatação.")
        return {"content": "Agente sem treinamento válido."}

def extract_text_from_pdf(pdf_path):
    try:
        return extract_text(pdf_path)
    except Exception as e:
        print(f"Erro ao extrair texto do PDF: {e}")
        return ""

def generate_markdown_report(content, output_path="analysis_report.md"):
    try:
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(f"# 📌 Análise de Business Model Canvas\n\n{content}\n")
        print(f"✅ Relatório em Markdown salvo como: {output_path}")
    except Exception as e:
        print(f"Erro ao gerar o arquivo Markdown: {e}")

def convert_md_to_pdf(md_path, pdf_path="analysis_report.pdf"):
    try:
        with open(md_path, "r", encoding="utf-8") as file:
            md_content = file.read()

        html_content = markdown.markdown(md_content)
        soup = BeautifulSoup(html_content, "html.parser")

        doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=60, bottomMargin=40)
        story = []
        
        styles = getSampleStyleSheet()
        
        title_style = ParagraphStyle(
            "Title",
            parent=styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.darkblue,
            spaceAfter=15,
            alignment=1,
        )

        section_title_style = ParagraphStyle(
            "SectionTitle",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            textColor=colors.darkgreen,
            spaceBefore=15,
            spaceAfter=10,
        )

        normal_style = ParagraphStyle(
            "Normal",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            spaceAfter=12,
        )

        bullet_style = ParagraphStyle(
            "Bullet",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=12,
            leading=16,
            leftIndent=20,
            bulletIndent=10,
            spaceAfter=8,
        )

        story.append(Paragraph("📌 Análise de Business Model Canvas", title_style))
        story.append(Spacer(1, 20))

        for elem in soup.find_all(["h1", "h2", "p", "ul"]):
            if elem.name == "h1":
                story.append(Paragraph(elem.get_text(), section_title_style))
                story.append(Spacer(1, 10))

            elif elem.name == "h2":
                story.append(Paragraph(elem.get_text(), section_title_style))
                story.append(Spacer(1, 8))

            elif elem.name == "p":
                story.append(Paragraph(elem.get_text(), normal_style))
                story.append(Spacer(1, 8))

            elif elem.name == "ul":
                items = [ListItem(Paragraph(li.get_text(), bullet_style)) for li in elem.find_all("li")]
                story.append(ListFlowable(items, bulletType="bullet"))

        def footer(canvas, doc):
            canvas.saveState()
            footer_text = "📄 Relatório gerado automaticamente - Aurora Tech"
            canvas.setFont("Helvetica", 9)
            canvas.drawRightString(540, 20, footer_text)
            canvas.restoreState()

        frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height - 40, id="normal")
        template = PageTemplate(id="report", frames=frame, onPage=footer)
        doc.addPageTemplates([template])

        doc.build(story)
        print(f"✅ PDF gerado com sucesso: {pdf_path}")

    except Exception as e:
        print(f"Erro ao converter Markdown para PDF: {e}")

def get_pdfs_from_folder(folder_path):
    try:
        return [f for f in os.listdir(folder_path) if f.endswith(".pdf")]
    except FileNotFoundError:
        print(f"⚠️ A pasta {folder_path} não foi encontrada!")
        return []

def correct_pdfs_in_folder(chat_model, chat_history, folder_path, output_folder):
    try:
        os.makedirs(output_folder, exist_ok=True)
        pdfs = get_pdfs_from_folder(folder_path)
        if not pdfs:
            print("⚠️ Nenhum PDF encontrado na pasta especificada.")
            return
        
        for pdf_file in pdfs:
            pdf_path = os.path.join(folder_path, pdf_file)
            print(f"📘 Analisando o PDF: {pdf_file}")
            
            extracted_text = extract_text_from_pdf(pdf_path)
            if extracted_text:
                print(f"Analisando conteúdo de {pdf_file}...")
                analysis_result = analyze_content_with_agent(chat_model, chat_history, extracted_text)
                
                md_path = f"analysis_report_{pdf_file.replace('.pdf', '.md')}"
                generate_markdown_report(analysis_result, md_path)

                corrected_pdf_path = os.path.join(output_folder, pdf_file.replace(".pdf", "_analysis.pdf"))
                convert_md_to_pdf(md_path, corrected_pdf_path)
                print(f"✅ Relatório gerado para {pdf_file}!\n")
            else:
                print(f"⚠️ Não foi possível extrair o texto de {pdf_file}.")
    except Exception as e:
        print(f"Erro ao corrigir os PDFs: {e}")

def initialize_agent():
    try:
        chat_history = load_history()
        training_data = load_training_data()

        if not any(msg.content == training_data["content"] for msg in chat_history):
            chat_history.insert(0, AIMessage(content=training_data["content"]))

        return chat_model, chat_history
    except Exception as e:
        print(f"Erro ao inicializar o agente: {e}")
        return chat_model, []

def analyze_content_with_agent(chat_model, chat_history, content):
    try:
        chat_history.append(HumanMessage(content=content))
        response = chat_model.invoke(chat_history)
        chat_history.append(AIMessage(content=response.content))
        return response.content
    except Exception as e:
        return f"Erro ao analisar conteúdo: {e}"

def main():
    chat_model, chat_history = initialize_agent()

    print("🤖 Chatbot Gemini - Análise de Business Model Canvas (digite 'sair' para encerrar)")

    folder_path = r"C:\Users\Gabriel Fernandes\Desktop\PROJETOS\Agent\LangChain\Gemini\pdfs"
    output_folder = r"C:\Users\Gabriel Fernandes\Desktop\PROJETOS\Agent\LangChain\Gemini\pdfs\correcoes"

    while True:
        user_input = input("Você: ")

        if user_input.lower() == "sair":
            print("Encerrando conversa. Histórico salvo!")
            save_history(chat_history)
            break

        if user_input.lower() == "corrigir pdfs":
            print(f"Iniciando correção de todos os PDFs na pasta '{folder_path}'...")
            correct_pdfs_in_folder(chat_model, chat_history, folder_path, output_folder)

        elif user_input.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(user_input)
            if extracted_text:
                print("📘 PDF carregado com sucesso!")
                analysis_result = analyze_content_with_agent(chat_model, chat_history, extracted_text)
                md_path = "analysis_report.md"
                generate_markdown_report(analysis_result, md_path)
                convert_md_to_pdf(md_path)
        else:
            chat_history.append(HumanMessage(content=user_input))
            response = chat_model.invoke(chat_history)
            print(f"Gemini: {response.content}\n")
            chat_history.append(AIMessage(content=response.content))

        save_history(chat_history)

if __name__ == "__main__":
    main()
