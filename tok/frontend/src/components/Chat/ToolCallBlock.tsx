import { useState } from 'react';
import {
  ChevronDown,
  ChevronRight,
  Terminal,
  FileText,
  Search,
  Globe,
  FolderSearch,
  Pencil,
  FilePlus,
  CheckCircle,
  XCircle,
  Loader,
} from 'lucide-react';
import clsx from 'clsx';
import type { ToolCall } from '@/types';

interface ToolCallBlockProps {
  toolCall: ToolCall;
  isStreaming?: boolean;
}

const TOOL_ICONS: Record<string, typeof Terminal> = {
  Bash: Terminal,
  Read: FileText,
  Grep: Search,
  WebSearch: Globe,
  WebFetch: Globe,
  Glob: FolderSearch,
  Edit: Pencil,
  Write: FilePlus,
};

export function ToolCallBlock({ toolCall, isStreaming = false }: ToolCallBlockProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const Icon = TOOL_ICONS[toolCall.name] || Terminal;

  // Format input for display
  const formatInput = () => {
    const input = toolCall.input_data;
    if (toolCall.name === 'Bash') {
      return input.command as string;
    }
    if (toolCall.name === 'Read') {
      return input.file_path as string;
    }
    if (toolCall.name === 'WebSearch') {
      return input.query as string;
    }
    if (toolCall.name === 'Grep') {
      return `${input.pattern} in ${input.path || '.'}`;
    }
    return JSON.stringify(input, null, 2);
  };

  return (
    <div className="tool-call rounded-lg overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full flex items-center gap-2 px-3 py-2 text-sm text-amber-300 hover:bg-amber-500/10 transition-colors"
      >
        <Icon size={16} className="text-amber-400" />
        <span className="font-medium">{toolCall.name}</span>
        <span className="flex-1 text-xs text-amber-400/70 text-left truncate font-mono">
          {formatInput()}
        </span>

        {/* Status indicator */}
        <span className="ml-2">
          {isStreaming ? (
            <Loader size={14} className="animate-spin text-amber-400" />
          ) : toolCall.is_error ? (
            <XCircle size={14} className="text-red-400" />
          ) : (
            <CheckCircle size={14} className="text-green-400" />
          )}
        </span>

        <span>
          {isExpanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        </span>
      </button>

      <div
        className={clsx(
          'overflow-hidden transition-all duration-200',
          isExpanded ? 'max-h-96' : 'max-h-0'
        )}
      >
        <div className="px-3 pb-3 space-y-2">
          {/* Input */}
          <div>
            <p className="text-xs text-amber-400/70 mb-1">Input:</p>
            <pre className="text-xs text-amber-200/80 bg-slate-800/50 p-2 rounded overflow-x-auto">
              {JSON.stringify(toolCall.input_data, null, 2)}
            </pre>
          </div>

          {/* Result */}
          {toolCall.result && (
            <div>
              <p className="text-xs text-amber-400/70 mb-1">
                {toolCall.is_error ? 'Error:' : 'Result:'}
              </p>
              <pre
                className={clsx(
                  'text-xs p-2 rounded overflow-x-auto max-h-48',
                  toolCall.is_error
                    ? 'text-red-300 bg-red-900/20'
                    : 'text-amber-200/80 bg-slate-800/50'
                )}
              >
                {toolCall.result}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
