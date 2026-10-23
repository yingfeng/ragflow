import { MessageType, WebSearchProvider } from '@/constants/chat';
import { IAttachment } from '@/hooks/use-send-message';

export interface IDocumentDownloadInfo {
  doc_id: string;
  filename: string;
  mime_type: string;
  size?: number;
}

export interface PromptConfig {
  empty_response: string;
  parameters: Parameter[];
  prologue: string;
  system: string;
  tts?: boolean;
  quote: boolean;
  keyword: boolean;
  refine_multiturn: boolean;
  use_kg: boolean;
  reasoning?: boolean;
  cross_languages?: Array<string>;
  brave_api_key?: string;
  exa_api_key?: string;
  firecrawl_api_key?: string;
  linkup_api_key?: string;
  parallel_api_key?: string;
  querit_api_key?: string;
  search1api_api_key?: string;
  serply_api_key?: string;
  tavily_api_key?: string;
  youcom_api_key?: string;
  web_search_provider?: WebSearchProvider;
  toc_enhance?: boolean;
}

export interface Parameter {
  key: string;
  optional: boolean;
}

/**
 * One structured step of an agent turn: a tool call, or a tool waiting for a
 * human. `request_id` addresses a pending approval — the same id comes back on
 * the verdict event, so a card is replaced rather than duplicated.
 */
export interface IThinkEvent {
  kind: string;
  stage?: string;
  tool?: string;
  args?: string;
  status?: string;
  reason?: string;
  cause?: string;
  results?: number;
  documents?: number;
  sources?: string[];
  duration_ms?: number;
  summary: string;
  request_id?: string;
}

/**
 * The limits one run of an agent works inside. Zero (or absent) means the limit
 * is off, so a field the editor left blank reads as "no limit" rather than "zero".
 */
export interface IHarnessAgentPolicy {
  max_rounds?: number;
  max_tokens?: number;
  max_tool_calls?: number;
  max_wall_sec?: number;
  approval?: 'auto' | 'ask';
}

/** What a user writes: the agent's instructions and its run limits. */
/** One skill the caller may attach to an agent, resolved from their skillspace. */
export interface IHarnessSkill {
  name: string;
  description: string;
  version?: string;
  when_to_use?: string;
  auto_load?: boolean;
  resources?: string[];
}

export interface IHarnessAgentDefinition {
  id?: string;
  name: string;
  prompt: string;
  policy?: IHarnessAgentPolicy;
  /** Skills selected from the caller's skillspace. Empty keeps the built-in set. */
  skills?: string[];
  [key: string]: unknown;
}

/** One stored version of an agent, as the editor sees it. */
export interface IHarnessAgentDetail {
  agent_id: string;
  version_id: string;
  version: number;
  name: string;
  definition: IHarnessAgentDefinition;
}

/** One pending tool approval as the harness reports it. */
export interface IHarnessApproval {
  request_id: string;
  run_id: string;
  tool: string;
  args: string;
  risk: string;
  tenant_id: string;
}

export interface LlmSetting {
  Creative: Variable;
  Custom: Variable;
  Evenly: Variable;
  Precise: Variable;
}

export interface Variable {
  frequency_penalty?: number;
  max_tokens?: number;
  presence_penalty?: number;
  temperature?: number;
  top_p?: number;
  thinking?: 'default' | 'enabled' | 'disabled';
  tenant_llm_id?: string;
  model_type?: string;
}

export interface IDialog {
  create_date: string;
  create_time: number;
  description: string;
  icon: string;
  id: string;
  dialog_id?: string;
  dataset_ids: string[];
  kb_names: string[];
  language: string;
  llm_id: string;
  tenant_llm_id?: string;
  llm_setting: Variable;
  llm_setting_type?: string;
  name: string;
  prompt_config: PromptConfig;
  prompt_type: string;
  status: string;
  tenant_id: string;
  update_date: string;
  update_time: number;
  keywords_similarity_weight?: number;
  similarity_threshold: number;
  top_k: number;
  top_n: number;
  rerank_candidates_count: number;
  rerank_id?: string;
  tenant_rerank_id?: string;
  meta_data_filter: MetaDataFilter;
  /** Set when this thread is run by an agent; empty for a plain chat. */
  agent_id?: string;
}

interface MetaDataFilter {
  manual: Manual[];
  method: string;
}

interface Manual {
  key: string;
  op: string;
  value: string;
}

export interface IConversation {
  create_date: string;
  create_time: number;
  chat_id: string;
  id: string;
  avatar: string;
  messages: Message[];
  reference: IReference[];
  name: string;
  update_date: string;
  update_time: number;
  is_new: true;
}

export interface Message {
  content: string;
  role: MessageType;
  doc_ids?: string[];
  prompt?: string;
  id?: string;
  audio_binary?: string;
  data?: any;
  files?: (File | UploadResponseDataType)[];
  chatBoxId?: string;
  attachment?: IAttachment;
  downloads?: IDocumentDownloadInfo[];
  /** Agent steps streamed with this turn, in arrival order. */
  think_events?: IThinkEvent[];
}

export interface IReferenceChunk {
  id: string;
  content: null;
  document_id: string;
  document_name: string;
  dataset_id: string;
  image_id: string;
  similarity: number;
  vector_similarity: number;
  term_similarity: number;
  positions: number[][];
  doc_type?: string;
}

export interface IReference {
  chunks: IReferenceChunk[];
  doc_aggs: Docagg[];
  total: number;
}

export interface IReferenceObject {
  chunks: Record<string, IReferenceChunk>;
  doc_aggs: Record<string, Docagg>;
}

export interface IAnswer {
  answer: string;
  attachment?: IAttachment;
  downloads?: IDocumentDownloadInfo[];
  reference?: IReference;
  conversationId?: string;
  prompt?: string;
  id?: string;
  audio_binary?: string;
  data?: any;
  chatBoxId?: string;
  think_events?: IThinkEvent[];
}

export interface Docagg {
  count: number;
  doc_id: string;
  doc_name: string;
  url?: string;
}

// interface Chunk {
//   chunk_id: string;
//   content_ltks: string;
//   content_with_weight: string;
//   doc_id: string;
//   docnm_kwd: string;
//   img_id: string;
//   important_kwd: any[];
//   kb_id: string;
//   similarity: number;
//   term_similarity: number;
//   vector_similarity: number;
// }

export interface IToken {
  create_date: string;
  create_time: number;
  tenant_id: string;
  token: string;
  update_date?: any;
  update_time?: any;
  beta: string;
}

export interface IStats {
  pv: [string, number][];
  uv: [string, number][];
  speed: [string, number][];
  tokens: [string, number][];
  round: [string, number][];
  thumb_up: [string, number][];
}

export interface IExternalChatInfo {
  avatar?: string;
  title: string;
  prologue?: string;
  has_tavily_key?: boolean;
  has_web_search_provider?: boolean;
  llm_id?: string;
}

export interface IMessage extends Message {
  id: string;
  awaitingServerId?: boolean;
  reference?: IReference; // the latest news has reference
  conversationId?: string; // To distinguish which conversation the message belongs to
}

export interface IClientConversation extends IConversation {
  messages: IMessage[];
}

export interface UploadResponseDataType {
  created_at: number;
  created_by: string;
  extension: string;
  id: string;
  mime_type: string;
  name: string;
  preview_url: null;
  size: number;
}
