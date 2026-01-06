import { useRef, useState, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { User, Bot, ChevronDown, ChevronRight, GitBranch } from 'lucide-react';
import { ThinkingBlock } from './ThinkingBlock';
import { ToolCallBlock } from './ToolCallBlock';
import type { Message as MessageType, TextSelection } from '@/types';
import clsx from 'clsx';

interface MessageProps {
  message: MessageType;
  branchId: string;
  onTextSelect: (selection: TextSelection | null) => void;
}

export function Message({ message, branchId, onTextSelect }: MessageProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [showBranchButton, setShowBranchButton] = useState(false);

  const handleMouseUp = useCallback(() => {
    if (message.role !== 'assistant') return;

    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !contentRef.current) {
      onTextSelect(null);
      setShowBranchButton(false);
      return;
    }

    const range = sel.getRangeAt(0);
    if (!contentRef.current.contains(range.commonAncestorContainer)) {
      onTextSelect(null);
      setShowBranchButton(false);
      return;
    }

    const text = sel.toString().trim();
    if (!text || text.length < 3) {
      onTextSelect(null);
      setShowBranchButton(false);
      return;
    }

    // Calculate offset
    const preSelectionRange = range.cloneRange();
    preSelectionRange.selectNodeContents(contentRef.current);
    preSelectionRange.setEnd(range.startContainer, range.startOffset);
    const start = preSelectionRange.toString().length;
    const end = start + text.length;

    onTextSelect({
      messageId: message.id,
      branchId,
      start,
      end,
      text,
    });
    setShowBranchButton(true);
  }, [message.id, message.role, branchId, onTextSelect]);

  const isUser = message.role === 'user';

  return (
    <div
      className={clsx(
        'flex gap-4',
        isUser ? 'flex-row-reverse' : ''
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center',
          isUser ? 'bg-blue-600' : 'bg-tok-600'
        )}
      >
        {isUser ? (
          <User size={18} className="text-white" />
        ) : (
          <Bot size={18} className="text-white" />
        )}
      </div>

      {/* Content */}
      <div className={clsx('flex-1 max-w-[85%]', isUser ? 'text-right' : '')}>
        {/* Thinking (collapsible) */}
        {message.thinking && (
          <ThinkingBlock thinking={message.thinking} />
        )}

        {/* Tool Calls */}
        {message.tool_calls.length > 0 && (
          <div className="space-y-2 mb-3">
            {message.tool_calls.map((tc) => (
              <ToolCallBlock key={tc.id} toolCall={tc} />
            ))}
          </div>
        )}

        {/* Message Content */}
        <div
          ref={contentRef}
          onMouseUp={handleMouseUp}
          className={clsx(
            'message-content rounded-2xl px-4 py-3',
            isUser
              ? 'bg-blue-600 text-white'
              : 'bg-slate-700 text-slate-100'
          )}
        >
          {isUser ? (
            <p className="whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div className="prose prose-invert prose-sm max-w-none">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {message.content}
              </ReactMarkdown>
            </div>
          )}
        </div>

        {/* Branch button hint */}
        {showBranchButton && !isUser && (
          <div className="mt-2 flex items-center gap-2 text-xs text-tok-400">
            <GitBranch size={12} />
            <span>Selection detected - a branch prompt will appear</span>
          </div>
        )}

        {/* Metadata */}
        <div
          className={clsx(
            'mt-1 text-xs text-slate-500',
            isUser ? 'text-right' : ''
          )}
        >
          {message.model && <span>{message.model}</span>}
        </div>
      </div>
    </div>
  );
}
