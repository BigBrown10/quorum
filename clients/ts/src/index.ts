export type ConsensusMode = "simple_majority" | "weighted_majority" | "quantum_ready";

export type AgentCandidate = {
  id: string;
  content: unknown;
  confidence?: number;
  sources?: string[];
  embedding?: number[];
  stats?: Record<string, unknown>;
};

export type ConsensusRequest = {
  candidates: AgentCandidate[];
  mode?: ConsensusMode;
};

export type DisagreementEdge = {
  source_id: string;
  target_id: string;
  weight: number;
};

export type ConsensusResponse = {
  consensus_answer: unknown;
  consensus_cluster_id: string;
  selected_agent_ids: string[];
  agreement_score: number;
  supporting_candidate_count: number;
  total_candidates: number;
  unstable: boolean;
  mode: ConsensusMode;
  disagreement_edges: DisagreementEdge[];
  rationale: string;
};

export class QuorumClient {
  constructor(private readonly baseUrl: string) {}

  async health(): Promise<{ status: string }> {
    const response = await fetch(new URL("/health", this.baseUrl));
    if (!response.ok) {
      throw new Error(`Quorum health check failed with status ${response.status}`);
    }

    return (await response.json()) as { status: string };
  }

  async resolveConsensus(request: ConsensusRequest): Promise<ConsensusResponse> {
    const response = await fetch(new URL("/resolve", this.baseUrl), {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Quorum resolve failed with status ${response.status}: ${detail}`);
    }

    return (await response.json()) as ConsensusResponse;
  }
}

export function createQuorumClient(baseUrl: string): QuorumClient {
  return new QuorumClient(baseUrl);
}
