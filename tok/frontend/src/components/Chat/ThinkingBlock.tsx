import { useState } from 'react';
import { Brain, ChevronDown, ChevronRight } from 'lucide-react';
import clsx from 'clsx';

interface ThinkingBlockProps {
  thinking: string;
  isStreaming?: boolean;
}

export function ThinkingBlock({ thinking, isStreaming = false }: ThinkingBlockProps) {
  const [isExpanded, setIsExpanded] = useState(isStreaming);

  // Auto-collapse when streaming ends
  if (!isStreaming && isExpanded && thinking.length > 500) {
    // Keep expanded for short thinking, collapse for long
  }

  return (
    <div className="thinking-block rounded-lg mb-3 overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-purple-300 hover:bg-purple-500/10 transition-colors"
      >
        <Brain size={16} className="text-purple-400" />
        <span className="font-medium">
          {isStreaming ? 'Thinking...' : 'Thinking'}
        </span>
        {isStreaming && (
          <span className="flex-1 text-xs text-purple-400/70 text-left truncate">
            {thinking.slice(-100)}...
          </span>
        )}
        <span className="ml-auto">
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </span>
      </button>

      <div
        className={clsx(
          'overflow-hidden transition-all duration-200',
          isExpanded ? 'max-h-96' : 'max-h-0'
        )}
      >
        <div className="px-3 pb-3 text-sm text-purple-200/80 whitespace-pre-wrap overflow-y-auto max-h-80">
          {thinking}
          {isStreaming && <span className="streaming-cursor" />}
        </div>
      </div>
    </div>
  );
}
