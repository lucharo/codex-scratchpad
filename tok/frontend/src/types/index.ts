// Tree of Knowledge Types

export interface Tree {
  id: string;
  name: string;
  created_at: string;
  updated_at: string;
  branch_count: number;
}

export interface BranchOrigin {
  source_message_id: string;
  source_branch_id: string;
  highlight_start: number;
  highlight_end: number;
  highlighted_text: string;
  user_prompt: string;
}

export interface Branch {
  id: string;
  tree_id: string;
  parent_branch_id: string | null;
  name: string;
  is_root: boolean;
  origin: BranchOrigin | null;
  message_count: number;
  created_at: string;
  messages?: Message[];
}

export interface ToolCall {
  id: string;
  tool_use_id: string;
  name: string;
  input_data: Record<string, unknown>;
  result: string | null;
  is_error: boolean;
}

export interface Attachment {
  id: string;
  filename: string;
  file_type: string;
  file_size: number;
}

export interface Message {
  id: string;
  branch_id: string;
  role: 'user' | 'assistant';
  content: string;
  thinking: string | null;
  position: number;
  model: string | null;
  tool_calls: ToolCall[];
  attachments: Attachment[];
  created_at: string;
}

export interface TreeDetail extends Tree {
  branches: Branch[];
}

// Streaming event types
export interface StreamEvent {
  type: 'text' | 'thinking' | 'tool_use' | 'tool_result' | 'complete' | 'error' | 'saved';
  content?: string;
  tool_name?: string;
  tool_input?: Record<string, unknown>;
  tool_result?: string;
  is_error?: boolean;
  session_id?: string;
  message_id?: string;
  model?: string;
}

// API request types
export interface CreateTreeRequest {
  name?: string;
}

export interface CreateBranchRequest {
  source_message_id: string;
  source_branch_id: string;
  highlight_start: number;
  highlight_end: number;
  highlighted_text: string;
  user_prompt: string;
  name?: string;
}

export interface SendMessageRequest {
  content: string;
}

// Text selection state
export interface TextSelection {
  messageId: string;
  branchId: string;
  start: number;
  end: number;
  text: string;
}
