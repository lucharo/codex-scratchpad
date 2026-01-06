import { useState, useCallback, useRef } from 'react';
import { sendMessageStream } from '@/lib/api';
import type { StreamEvent, ToolCall } from '@/types';

interface StreamingState {
  isStreaming: boolean;
  text: string;
  thinking: string;
  toolCalls: ToolCall[];
  error: string | null;
  sessionId: string | null;
  savedMessageId: string | null;
}

const initialState: StreamingState = {
  isStreaming: false,
  text: '',
  thinking: '',
  toolCalls: [],
  error: null,
  sessionId: null,
  savedMessageId: null,
};

export function useStreaming() {
  const [state, setState] = useState<StreamingState>(initialState);
  const abortRef = useRef<AbortController | null>(null);

  const sendMessage = useCallback(
    async (treeId: string, branchId: string, content: string) => {
      // Reset state
      setState({ ...initialState, isStreaming: true });

      try {
        const toolCalls: ToolCall[] = [];
        let currentToolCall: Partial<ToolCall> | null = null;

        for await (const event of sendMessageStream(treeId, branchId, { content })) {
          switch (event.type) {
            case 'text':
              setState((prev) => ({
                ...prev,
                text: prev.text + (event.content || ''),
              }));
              break;

            case 'thinking':
              setState((prev) => ({
                ...prev,
                thinking: prev.thinking + (event.content || ''),
              }));
              break;

            case 'tool_use':
              currentToolCall = {
                id: `temp_${Date.now()}`,
                tool_use_id: `tool_${Date.now()}`,
                name: event.tool_name || 'unknown',
                input_data: event.tool_input || {},
                result: null,
                is_error: false,
              };
              toolCalls.push(currentToolCall as ToolCall);
              setState((prev) => ({
                ...prev,
                toolCalls: [...toolCalls],
              }));
              break;

            case 'tool_result':
              if (currentToolCall) {
                currentToolCall.result = event.tool_result || null;
                currentToolCall.is_error = event.is_error || false;
                setState((prev) => ({
                  ...prev,
                  toolCalls: [...toolCalls],
                }));
              }
              currentToolCall = null;
              break;

            case 'complete':
              setState((prev) => ({
                ...prev,
                sessionId: event.session_id || null,
              }));
              break;

            case 'saved':
              setState((prev) => ({
                ...prev,
                savedMessageId: event.message_id || null,
                isStreaming: false,
              }));
              break;

            case 'error':
              setState((prev) => ({
                ...prev,
                error: event.content || 'Unknown error',
                isStreaming: false,
              }));
              break;
          }
        }
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: err instanceof Error ? err.message : 'Unknown error',
          isStreaming: false,
        }));
      }
    },
    []
  );

  const reset = useCallback(() => {
    setState(initialState);
  }, []);

  const abort = useCallback(() => {
    abortRef.current?.abort();
    setState((prev) => ({ ...prev, isStreaming: false }));
  }, []);

  return {
    ...state,
    sendMessage,
    reset,
    abort,
  };
}
