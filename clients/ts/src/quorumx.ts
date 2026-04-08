export type QuorumXConsensusMode =
  | "simple_majority"
  | "weighted_majority"
  | "graph_min_cut"
  | "quantum_ready";

export type QuorumXMessage = {
  role: string;
  content: unknown;
  [key: string]: unknown;
};

export type QuorumXConfig = {
  n_agents?: number;
  max_rounds?: number;
  stability_threshold?: number;
  token_cap_per_agent_round?: number;
  consensus_mode?: QuorumXConsensusMode;
  system_instructions?: string;
  roles?: string[];
  model?: string;
  quorum_model?: string | null;
  temperature?: number;
  backend?: string | null;
  api_key?: string | null;
  base_url?: string | null;
  request_timeout_seconds?: number;
  mock_mode?: boolean;
};

export type QuorumXRequest = {
  task: string;
  system_instructions?: string;
  messages?: QuorumXMessage[];
  roles?: string[];
  config?: QuorumXConfig;
};

export type QuorumXBenchmark = {
  agent_id: string;
  stance: string;
  rounds_used: number;
  token_count: number;
  confidence: number;
  answer_preview: string;
};

export type QuorumXResult = {
  answer: unknown;
  agreement_score: number;
  unstable: boolean;
  rounds_used: number;
  total_tokens: number;
  prompt_tokens: number;
  completion_tokens: number;
  tokens_per_round: number[];
  benchmark: QuorumXBenchmark[];
  disagreement_edges_final: Array<{ source_id: string; target_id: string; weight: number }>;
  selected_agent_ids: string[];
  consensus_mode: string;
  rationale: string;
};

export type QuorumXChatRequest = {
  messages: QuorumXMessage[];
  system_instructions?: string;
  roles?: string[];
  config?: QuorumXConfig;
};

export type QuorumXChatResponse = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: { role: "assistant"; content: unknown };
    finish_reason: "stop";
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  quorumx: QuorumXResult;
};

export class QuorumXClient {
  constructor(private readonly baseUrl: string) {}

  async run(request: QuorumXRequest): Promise<QuorumXResult> {
    const response = await fetch(new URL("/v1/quorumx", this.baseUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`QuorumX resolve failed with status ${response.status}: ${detail}`);
    }

    return (await response.json()) as QuorumXResult;
  }

  async chatCompletions(request: QuorumXChatRequest): Promise<QuorumXChatResponse> {
    const response = await fetch(new URL("/v1/chat/completions", this.baseUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`QuorumX chat completions failed with status ${response.status}: ${detail}`);
    }

    return (await response.json()) as QuorumXChatResponse;
  }
}

export function createQuorumXClient(baseUrl: string): QuorumXClient {
  return new QuorumXClient(baseUrl);
}
