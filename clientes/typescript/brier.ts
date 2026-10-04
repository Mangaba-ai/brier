// Cliente TypeScript do Brier (fetch nativo: Node 18+, Deno, Bun e navegadores).
//   const b = new Brier("http://127.0.0.1:8790");
//   const r = await b.decide("internet caiu de novo", { setor: { type: "choice",
//     instructions: "Qual time atende?", criteria: { suporte: "Queda", retencao: "Cancelamento" } } });

export type Pergunta =
  | { type: "choice"; instructions: string | object; criteria: Record<string, string | object> }
  | { type: "score"; instructions: string | object; criteria: (string | object)[] }
  | { type: "noul"; instructions: string | object; criteria: { true: string | object; false: string | object } };

export interface Resposta {
  type: "choice" | "score" | "noul";
  choice?: string;
  score?: number;
  noul?: number;
  probabilities?: Record<string, number>;
  confidence?: number;
  low_confidence?: boolean;
}

export interface Decisao {
  answers: Record<string, Resposta>;
  model: string;
  latencia_ms: number;
  roteamento?: { idioma: string; modelo: string };
  abstencao?: { min_confidence: number; perguntas: string[] };
  avisos?: string[];
}

export class Brier {
  constructor(private url = "http://127.0.0.1:8790", private timeoutMs = 30000) {}

  private async post<T>(caminho: string, corpo: unknown): Promise<T> {
    const r = await fetch(this.url + caminho, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(corpo),
      signal: AbortSignal.timeout(this.timeoutMs),
    });
    const dados = await r.json();
    if (!r.ok) throw new Error(`Brier ${r.status}: ${JSON.stringify(dados.detail ?? dados)}`);
    return dados as T;
  }

  decide(state: unknown, questions: Record<string, Pergunta>, opcoes: { min_confidence?: number; longo?: "auto" | "sim" | "nao" } = {}) {
    return this.post<Decisao>("/v1/decide", { state, questions, ...opcoes });
  }

  async decideLote(states: unknown[], questions: Record<string, Pergunta>, opcoes: { min_confidence?: number } = {}) {
    return (await this.post<{ resultados: Decisao[] }>("/v1/decide/lote", { states, questions, ...opcoes })).resultados;
  }
}
