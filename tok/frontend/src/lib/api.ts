// API client for Tree of Knowledge

import type {
  Tree,
  TreeDetail,
  Branch,
  Message,
  CreateTreeRequest,
  CreateBranchRequest,
  SendMessageRequest,
  StreamEvent,
} from '@/types';

const API_BASE = '/api';

async function fetchJSON<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${url}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options?.headers,
    },
    ...options,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

// Trees
export async function listTrees(): Promise<Tree[]> {
  return fetchJSON('/trees');
}

export async function createTree(data: CreateTreeRequest = {}): Promise<Tree> {
  return fetchJSON('/trees', {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

export async function getTree(treeId: string): Promise<TreeDetail> {
  return fetchJSON(`/trees/${treeId}`);
}

export async function updateTree(treeId: string, data: { name: string }): Promise<Tree> {
  return fetchJSON(`/trees/${treeId}`, {
    method: 'PATCH',
    body: JSON.stringify(data),
  });
}

export async function deleteTree(treeId: string): Promise<void> {
  await fetchJSON(`/trees/${treeId}`, { method: 'DELETE' });
}

// Branches
export async function listBranches(treeId: string): Promise<Branch[]> {
  return fetchJSON(`/trees/${treeId}/branches`);
}

export async function getBranch(treeId: string, branchId: string): Promise<Branch> {
  return fetchJSON(`/trees/${treeId}/branches/${branchId}`);
}

export async function createBranch(treeId: string, data: CreateBranchRequest): Promise<Branch> {
  return fetchJSON(`/trees/${treeId}/branches`, {
    method: 'POST',
    body: JSON.stringify(data),
  });
}

// Messages
export async function listMessages(treeId: string, branchId: string): Promise<Message[]> {
  return fetchJSON(`/trees/${treeId}/branches/${branchId}/messages`);
}

export async function getMessage(
  treeId: string,
  branchId: string,
  messageId: string
): Promise<Message> {
  return fetchJSON(`/trees/${treeId}/branches/${branchId}/messages/${messageId}`);
}

// Streaming message sending
export async function* sendMessageStream(
  treeId: string,
  branchId: string,
  data: SendMessageRequest
): AsyncGenerator<StreamEvent> {
  const response = await fetch(
    `${API_BASE}/trees/${treeId}/branches/${branchId}/messages`,
    {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    }
  );

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  const reader = response.body?.getReader();
  if (!reader) {
    throw new Error('No response body');
  }

  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6));
          yield data as StreamEvent;
        } catch {
          // Skip malformed JSON
        }
      }
    }
  }

  // Process remaining buffer
  if (buffer.startsWith('data: ')) {
    try {
      const data = JSON.parse(buffer.slice(6));
      yield data as StreamEvent;
    } catch {
      // Skip malformed JSON
    }
  }
}

// File upload
export async function uploadFile(file: File, messageId?: string): Promise<{ id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  if (messageId) {
    formData.append('message_id', messageId);
  }

  const response = await fetch(`${API_BASE}/upload`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
    throw new Error(error.detail);
  }

  return response.json();
}
