import { useState, useRef, useEffect } from 'react';
import { GitBranch, X, Send } from 'lucide-react';
import clsx from 'clsx';

interface BranchPromptProps {
  selectedText: string;
  onSubmit: (prompt: string) => void;
  onCancel: () => void;
}

export function BranchPrompt({ selectedText, onSubmit, onCancel }: BranchPromptProps) {
  const [prompt, setPrompt] = useState('');
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!prompt.trim()) return;
    onSubmit(prompt.trim());
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
    if (e.key === 'Escape') {
      onCancel();
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <div className="bg-slate-800 rounded-xl shadow-2xl max-w-lg w-full">
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-3 border-b border-slate-700">
          <div className="flex items-center gap-2 text-tok-400">
            <GitBranch size={18} />
            <span className="font-medium">Create Branch</span>
          </div>
          <button
            onClick={onCancel}
            className="p-1 hover:bg-slate-700 rounded transition-colors"
          >
            <X size={18} className="text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-4">
          {/* Selected text preview */}
          <div className="mb-4">
            <p className="text-xs text-slate-400 mb-2">Selected text:</p>
            <blockquote className="text-sm text-slate-300 italic border-l-2 border-tok-500 pl-3 py-1 bg-slate-700/50 rounded-r">
              "{selectedText.length > 200
                ? `${selectedText.slice(0, 200)}...`
                : selectedText}"
            </blockquote>
          </div>

          {/* Prompt input */}
          <form onSubmit={handleSubmit}>
            <label className="block text-sm text-slate-400 mb-2">
              What would you like to explore about this?
            </label>
            <textarea
              ref={inputRef}
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="e.g., Can you explain this in more detail?"
              rows={3}
              className={clsx(
                'w-full px-4 py-3 bg-slate-700 rounded-lg resize-none',
                'text-white placeholder-slate-500',
                'focus:outline-none focus:ring-2 focus:ring-tok-500'
              )}
            />

            <div className="flex justify-end gap-3 mt-4">
              <button
                type="button"
                onClick={onCancel}
                className="px-4 py-2 text-slate-400 hover:text-white transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!prompt.trim()}
                className={clsx(
                  'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
                  prompt.trim()
                    ? 'bg-tok-600 text-white hover:bg-tok-700'
                    : 'bg-slate-700 text-slate-500 cursor-not-allowed'
                )}
              >
                <GitBranch size={16} />
                <span>Create Branch</span>
              </button>
            </div>
          </form>
        </div>

        {/* Help text */}
        <div className="px-4 py-3 border-t border-slate-700 bg-slate-800/50 rounded-b-xl">
          <p className="text-xs text-slate-500">
            A new conversation branch will be created with full context up to this
            point, plus your highlighted text and question.
          </p>
        </div>
      </div>
    </div>
  );
}
