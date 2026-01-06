import { Message } from './Message';
import { StreamingMessage } from './StreamingMessage';
import type { Message as MessageType, ToolCall, TextSelection } from '@/types';

interface MessageListProps {
  messages: MessageType[];
  branchId: string;
  streamingText: string | null;
  streamingThinking: string | null;
  streamingToolCalls: ToolCall[];
  onTextSelect: (selection: TextSelection | null) => void;
}

export function MessageList({
  messages,
  branchId,
  streamingText,
  streamingThinking,
  streamingToolCalls,
  onTextSelect,
}: MessageListProps) {
  return (
    <div className="max-w-4xl mx-auto px-4 py-6 space-y-6">
      {messages.length === 0 && !streamingText && (
        <div className="text-center py-12 text-slate-500">
          <p className="text-lg mb-2">Start a conversation</p>
          <p className="text-sm">
            Type a message below or ask Claude anything
          </p>
        </div>
      )}

      {messages.map((message) => (
        <Message
          key={message.id}
          message={message}
          branchId={branchId}
          onTextSelect={onTextSelect}
        />
      ))}

      {streamingText !== null && (
        <StreamingMessage
          text={streamingText}
          thinking={streamingThinking}
          toolCalls={streamingToolCalls}
        />
      )}
    </div>
  );
}
