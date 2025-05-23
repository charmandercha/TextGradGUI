import gradio as gr
import textgrad as tg
import requests
import os
import logging
from typing import Tuple, List, Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do Ollama
OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "qwen3:4b" # Um bom modelo padrão para geração e crítica

# --- Funções de Conexão com Ollama (sem alterações) ---
def validate_ollama_connection() -> Tuple[bool, List[str]]:
    """Valida conexão com Ollama e retorna modelos disponíveis"""
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = sorted(list(set([model["name"] for model in models])))
            return True, model_names
        return False, []
    except requests.exceptions.RequestException:
        return False, []

def get_available_models() -> List[str]:
    """Recupera modelos disponíveis no Ollama"""
    is_connected, models = validate_ollama_connection()
    return models if is_connected and models else [DEFAULT_MODEL, "qwen3", "llama3.1"]

# --- Lógica Principal de Otimização (Refatorada) ---

def run_optimization_flow(
    question: str,
    optimization_criteria: str,
    iterations: int,
    model_name: str
) -> Tuple[str, str, List[Dict]]:
    """
    Executa um fluxo completo de geração e otimização com TextGrad e Ollama.
    """
    # 1. Validação das Entradas do Usuário
    if not question.strip() or len(question) < 10:
        error_msg = "A Pergunta é obrigatória (mínimo 10 caracteres)."
        history = [{"iteration": 0, "error": error_msg, "status": "validation_failed"}]
        return "Erro de validação", "Falha", history
    if not optimization_criteria.strip():
        error_msg = "Os Critérios de Melhoria são obrigatórios."
        history = [{"iteration": 0, "error": error_msg, "status": "validation_failed"}]
        return "Erro de validação", "Falha", history

    try:
        # 2. Configurar o Engine TextGrad para usar Ollama via LiteLLM
        # NOTA: Usamos override=True para permitir que o engine seja reconfigurado
        # a cada execução sem causar erro.
        model_path = f"ollama/{model_name}"
        tg.set_backward_engine(f"experimental:{model_path}", override=True, cache=False)
        logger.info(f"Engine de otimização configurado para o modelo: {model_name}")

        # 3. Definir Variáveis TextGrad
        question_var = tg.Variable(
            question,
            role_description="A pergunta do usuário que precisa ser respondida.",
            requires_grad=False
        )

        # 4. GERAÇÃO DA RESPOSTA INICIAL (Novo passo crucial para UX)
        logger.info("Gerando a resposta inicial...")
        # Usamos BlackboxLLM para que o próprio modelo crie a primeira resposta.
        initial_model = tg.BlackboxLLM(f"experimental:{model_path}")
        answer_var = initial_model(question_var)
        # É importante definir a descrição do papel da variável que será otimizada.
        answer_var.set_role_description("Uma resposta que será otimizada iterativamente.")
        
        initial_answer_text = answer_var.value
        logger.info(f"Resposta inicial gerada: {initial_answer_text[:150]}...")

        history = [{
            "iteration": 0,
            "answer": initial_answer_text,
            "feedback": "Resposta inicial gerada pelo modelo.",
            "status": "initial"
        }]

        # 5. Configurar o Sistema de Otimização
        # NOTA: Usando 'lr' pois sua versão do textgrad espera este argumento.
        # Versões mais novas podem esperar 'learning_rate'.
        optimizer = tg.TGD(parameters=[answer_var])
        loss_fn = tg.TextLoss(optimization_criteria)

        # 6. Loop de Otimização
        logger.info(f"Iniciando {iterations} iterações de otimização...")
        for i in range(iterations):
            try:
                logger.info(f"Executando iteração {i + 1}/{iterations}")
                
                # Calcula a "perda" (o feedback textual)
                loss = loss_fn(answer_var)
                
                # Calcula o "gradiente" (a instrução de melhoria)
                loss.backward()
                
                # Aplica a otimização
                optimizer.step()
                
                # Limpa o gradiente para a próxima iteração
                optimizer.zero_grad()
                
                # Salva o progresso
                history.append({
                    "iteration": i + 1,
                    "answer": answer_var.value,
                    "feedback": loss.value, # O feedback é o valor da "loss"
                    "status": "success"
                })
                logger.info(f"Iteração {i + 1} concluída.")

            except Exception as step_error:
                error_msg = f"Erro na iteração {i + 1}: {step_error}"
                logger.error(error_msg)
                history.append({"iteration": i + 1, "error": error_msg, "status": "failed"})
                # Para o processo se uma iteração falhar
                break

        final_answer = history[-1].get("answer", initial_answer_text)
        final_status = f"Concluído após {len(history) - 1} iterações."
        return final_answer, final_status, history

    except Exception as e:
        error_msg = f"Erro geral no sistema: {e}"
        logger.error(error_msg, exc_info=True)
        history = [{"iteration": 0, "error": error_msg, "status": "system_error"}]
        return "Ocorreu um erro no sistema.", "Falha", history


# --- Interface Gráfica com Gradio (Refatorada para melhor UX) ---

with gr.Blocks(title="TextGrad Optimizer com Ollama", theme=gr.themes.Soft()) as interface:
    gr.Markdown("""
    # 🚀 Otimizador de Textos com TextGrad & Ollama
    Escreva uma pergunta e defina como a resposta deve ser melhorada. A IA irá gerar uma resposta inicial e depois a otimizará iterativamente seguindo seus critérios.
    """)

    with gr.Row():
        connection_status = gr.HTML()

    with gr.Row(variant="panel"):
        with gr.Column(scale=2):
            gr.Markdown("### 1. Defina a Tarefa")
            
            question_input = gr.Textbox(
                label="Pergunta ou Instrução",
                placeholder="Ex: Explique o que é a Teoria da Relatividade.",
                lines=4
            )

            optimization_criteria = gr.Textbox(
                label="Critérios para Melhoria (Loss Function)",
                info="Como a resposta deve ser avaliada e melhorada a cada passo?",
                value="Seja muito crítico. Avalie a clareza, precisão e simplicidade da resposta. Aponte erros factuais e sugira formas de tornar a explicação mais fácil de entender.",
                lines=5
            )

            gr.Markdown("### 2. Configure a Otimização")
            
            model_dropdown = gr.Dropdown(
                choices=get_available_models(),
                value=DEFAULT_MODEL if DEFAULT_MODEL in get_available_models() else (get_available_models()[0] if get_available_models() else ""),
                label="Modelo Ollama",
                info="Modelo usado para gerar e otimizar a resposta."
            )
            
            iterations = gr.Slider(
                1, 5, value=2, step=1,
                label="Ciclos de Otimização",
                info="Quantas vezes a IA deve tentar melhorar a própria resposta."
            )
            
            with gr.Row():
                optimize_btn = gr.Button("🧠 Gerar e Otimizar", variant="primary", scale=3)
                clear_btn = gr.Button("🗑️ Limpar", scale=1)

        with gr.Column(scale=3):
            gr.Markdown("### 3. Acompanhe o Resultado")
            
            with gr.Tab("✅ Resposta Final"):
                optimized_answer = gr.Textbox(
                    label="Resposta Otimizada",
                    lines=15,
                    interactive=False,
                    show_copy_button=True
                )
                final_status = gr.Textbox(label="Status Final", interactive=False)
                
            with gr.Tab("📜 Histórico Detalhado"):
                history_display = gr.JSON(label="Log de Iterações")

    # --- Lógica de Eventos da Interface ---

    def check_connection_status():
        is_connected, models = validate_ollama_connection()
        if is_connected:
            return f"<p style='color:green;'>✅ Ollama conectado | Modelos disponíveis: {len(models)}</p>"
        return f"<p style='color:red;'>❌ Ollama não detectado em {OLLAMA_BASE_URL}. Verifique se está em execução.</p>"

    # Ao carregar a interface, verifica a conexão
    interface.load(fn=check_connection_status, outputs=[connection_status])
    
    # Ação do botão de otimizar
    optimize_btn.click(
        fn=run_optimization_flow,
        inputs=[question_input, optimization_criteria, iterations, model_dropdown],
        outputs=[optimized_answer, final_status, history_display]
    )
    
    # Ação do botão de limpar
    clear_btn.click(
        fn=lambda: ("", "Avalie a clareza, precisão e simplicidade...", 2, "", "Aguardando otimização...", None),
        outputs=[question_input, optimization_criteria, iterations, optimized_answer, final_status, history_display]
    )

if __name__ == "__main__":
    is_connected, models = validate_ollama_connection()
    if is_connected:
        print("✅ Ollama detectado!")
        print(f"📦 Modelos disponíveis: {models}")
        print("🚀 Iniciando TextGrad Optimizer...")
        interface.launch(server_name="127.0.0.1", server_port=7860)
    else:
        print("❌ ERRO: Ollama não foi detectado ou não está acessível.")
        print(f"   Por favor, garanta que o Ollama esteja em execução e acessível em '{OLLAMA_BASE_URL}'.")