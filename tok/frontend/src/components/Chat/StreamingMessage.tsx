import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Bot } from 'lucide-react';
import { ThinkingBlock } from './ThinkingBlock';
import { ToolCallBlock } from './ToolCallBlock';
import type { ToolCall } from '@/types';

interface StreamingMessageProps {
  text: string;
  thinking: string | null;
  toolCalls: ToolCall[];
}

export function StreamingMessage({ text, thinking, toolCalls }: StreamingMessageProps) {
  return (
    <div className="flex gap-4">
      {/* Avatar */}
      <div className="flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center bg-tok-600">
        <Bot size={18} className="text-white" />
      </div>

      {/* Content */}
      <div className="flex-1 max-w-[85%]">
        {/* Thinking (live) */}
        {thinking && (
          <ThinkingBlock thinking={thinking} isStreaming />
        )}

        {/* Tool Calls (live) */}
        {toolCalls.length > 0 && (
          <div className="space-y-2 mb-3">
            {toolCalls.map((tc, idx) => (
              <ToolCallBlock
                key={tc.id || idx}
                toolCall={tc}
                isStreaming={tc.result === null}
              />
            ))}
          </div>
        )}

        {/* Message Content */}
        <div className="message-content rounded-2xl px-4 py-3 bg-slate-700 text-slate-100">
          {text ? (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {text}
              </ReactMarkdown>
              <span className="streaming-cursor" />
            </div>
          ) : (
            <div className="flex items-center gap-2 text-slate-400">
              <div className="w-2 h-2 bg-tok-500 rounded-full animate-pulse" />
              <span>Thinking...</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
