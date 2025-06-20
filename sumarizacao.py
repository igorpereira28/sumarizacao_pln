from transformers import pipeline
import re

def contar_estatisticas(texto):
    """Calcula caracteres, palavras e letras"""
    caracteres = len(texto)
    palavras = len(texto.split())
    letras = sum(c.isalpha() for c in texto)
    return caracteres, palavras, letras

def criar_resumo_claro(texto):
    """Gera um resumo claro e coerente em português"""
    try:
        # Usando o modelo BART para sumarização
        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            tokenizer="facebook/bart-large-cnn"
        )

        # Pré-processamento
        texto = re.sub(r'\s+', ' ', texto).strip()

        # Divisão em partes para textos longos
        max_chunk = 1000
        partes = [texto[i:i+max_chunk] for i in range(0, len(texto), max_chunk)]

        resumos = []
        for parte in partes:
            resumo = summarizer(
                parte,
                max_length=120,
                min_length=40,
                do_sample=False,
                num_beams=4,
                repetition_penalty=2.2,
                no_repeat_ngram_size=3,
                truncation=True
            )
            resumos.append(resumo[0]['summary_text'])

        # Combinação e limpeza
        resumo_final = ' '.join(resumos)
        resumo_final = re.sub(r'\.\s+\.', '.', resumo_final)  # Remove pontos duplos
        resumo_final = '. '.join(sent.capitalize() for sent in resumo_final.split('. ') if sent.strip())

        return resumo_final

    except Exception as e:
        print(f"Erro durante a sumarização: {str(e)}")
        return "Não foi possível gerar o resumo"

# Texto de entrada
texto = '''
Nos últimos anos, o calendário do futebol nacional vem sido discutido para ter modificação na fórmula de torneios estaduais, (que tem uma importância no contexto histórico do futebol brasileiro), regionais e nacionais, a fim de que todos os clubes possam ter um calendário regular. Os times grandes têm um calendário inchado, ou seja, um ano inteiro, ao contrário dos pequenos que tem pouco menos de 3 meses, logo após dispensam jogadores, comissão técnica e entre outros. E fecham suas portas.

Desde que o futebol começou no país, os clubes iam sendo fundados e entrando nos campeonatos estaduais, que na época tinha de grande importância e não existia campeonato nacionais. Mas, tinha as competições interestaduais, como a Taça Rio-São Paulo (extinta em 2002). Os clubes não tinham muitos jogos oficiais, e faziam excursões, pelo estado, país ou pelo mundo.

Ao longo do tempo com a criação dos campeonatos nacionais em 1959, os clubes foram se adequando, em 1971   campeonato brasileiro foi criado e desde então foi sendo modificando a fórmula a cada ano. Somente em 2003 é que o Brasileirão adotou a fórmula de pontos corridos: todos contra todos, que até hoje é utilizado. Também foi adotado na segunda divisão em 2006, e houve modificações em 2009 na terceira divisão.

Com o passar dos anos foi criada mais uma divisão do futebol nacional, como a Série D em 2009, a quarta divisão, que regularizou calendário de 40 equipes menores, atualmente ampliada para 68. O processo de classificação utilizada qualificar para a competição é pelo dos estaduais. Mas com a abertura da nova divisão foi pouco porque só 15% dos clubes ainda têm um calendário para o ano.

O calendário atual não atende a lógica do calendário internacional ao promover jogos de clubes nacionais em data FIFA (Fedération Internationale de Football Association) como por exemplo, uma rodada de campeonato brasileiro no mesmo dia ou entre o período de Data FIFA em que a Seleção Brasileira joga, por isso alguns clubes jogam sem seus principais jogadores. (Data FIFA são jogos das seleções nacionais) (TRIVELA, 2017).

Por isso, todo ano tem esse grande debate: a reformulação do calendário, O que se pensar? O que fazer? Como modificar? Então, foi fundado em 2013 o movimento “Bom Senso FC”, com 5 pontos objetivos, um deles é o ajuste do atual calendário nacional, esse movimento perdeu força em 2016 e acabou extinto, mas a discussão continua. A proposta era da criação da Série E, quinta divisão regionalizada, que incluiria cerca 432 clubes, promovendo a mudança em todo o calendário nacional. Assim como, favoreceria a todos os clubes menores do país que tivessem uma temporada completa, sem se preocupar com a escassez de jogos, dispensa de jogadores e comissão técnica, falta de público, problemas financeiros e a falta de visibilidade de patrocinadores.  Os dois últimos são uma das causas do atraso de salários dos jogadores (BOMSENSO, 2017).

Para os jogadores, seria de grande importância a mudança do calendário, por conta do atraso de salários, por falta de dinheiro dos clubes e o desemprego que atinge cerca de 20 mil profissionais. Muitos jogadores mantêm o sonho de ser “jogador de futebol”, como alguns boleiros que jogam pelo dinheiro e fama, bem como para dar uma vida melhor às famílias ou porque amam futebol ou, talvez, poder todas as essas opções.

Em minha opinião, concordo com o extinto “Bom Senso FC” pela mudança do regulamento e a estrutura do calendário do futebol brasileiro. Pois, favoreceria todos os clubes do país, seja Santos, Corinthians, Criciúma e outros times grandes. Os números de jogos são em média de 50, 60, 70 jogos. Alguns times, como Atlético Tubarão, Novo Hamburgo e etc., jogam em média 10, 20, 30, ou menos que isso.

Dessa foram, para que não se tenha problemas, com salários, escassez de jogos e outros já citados, a melhor forma é a criação de uma nova divisão, ou seja, uma quinta divisão. Isso não significa o fim dos tradicionais campeonatos estaduais. Apesar de alguns jornalistas acharem que com os estaduais deveriam acabar. Para eles, o estadual não traz muita emoção e interesse como era antigamente, no entanto eles cobram dos times grandes o título anual, caso não ganhem a mídia divulga que o time está crise. Isso gera uma grande polêmica. Não sou a favor do fim dos estaduais, só que eles sejam mudados e se tornem divisões inferiores, assim os times grandes não jogariam os estaduais. Esses torneios só se ajustariam no calendário, isso causa uma grande discussão todos os anos, uma discussão utópica! Mas, que é sempre debatida com esperança de uma modificação no futebol nacional, haja vista que futuramente poderá acontecer! Todavia, é preciso ter paciência com a autoridade máxima do futebol, afinal “Futebol não é só um jogo”!!
'''

# Calcula estatísticas do texto original
chars_orig, palavras_orig, letras_orig = contar_estatisticas(texto)

# Gera o resumo
resumo = criar_resumo_claro(texto)

# Calcula estatísticas do resumo
chars_resumo, palavras_resumo, letras_resumo = contar_estatisticas(resumo)

# Exibe os resultados
print("\n=== ESTATÍSTICAS ===")
print(f"Texto Original:")
print(f"- Caracteres: {chars_orig}")
print(f"- Palavras: {palavras_orig}")
print(f"- Letras: {letras_orig}\n")

print(f"Resumo Gerado:")
print(f"- Caracteres: {chars_resumo}")
print(f"- Palavras: {palavras_resumo}")
print(f"- Letras: {letras_resumo}\n")

print("=== TEXTO ORIGINAL ===")
print(texto)

print("=== RESUMO ===")
print(resumo)