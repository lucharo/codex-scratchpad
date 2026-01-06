import { useState, useRef, useCallback } from 'react';
import { Send, Paperclip, Image, Loader } from 'lucide-react';
import clsx from 'clsx';

interface ChatInputProps {
  onSend: (content: string, files?: File[]) => void;
  disabled?: boolean;
  placeholder?: string;
}

export function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Send a message...',
}: ChatInputProps) {
  const [content, setContent] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (!content.trim() && files.length === 0) return;
      if (disabled) return;

      onSend(content.trim(), files.length > 0 ? files : undefined);
      setContent('');
      setFiles([]);
    },
    [content, files, disabled, onSend]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        handleSubmit(e);
      }
    },
    [handleSubmit]
  );

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    setFiles((prev) => [...prev, ...selectedFiles]);
    e.target.value = ''; // Reset input
  }, []);

  const removeFile = useCallback((index: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== index));
  }, []);

  // Auto-resize textarea
  const handleInput = useCallback(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, []);

  return (
    <div className="border-t border-slate-700 bg-slate-800/50 p-4">
      {/* File previews */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-2 mb-3">
          {files.map((file, index) => (
            <div
              key={index}
              className="flex items-center gap-2 px-3 py-1.5 bg-slate-700 rounded-lg text-sm"
            >
              {file.type.startsWith('image/') ? (
                <Image size={14} className="text-blue-400" />
              ) : (
                <Paperclip size={14} className="text-slate-400" />
              )}
              <span className="max-w-32 truncate">{file.name}</span>
              <button
                onClick={() => removeFile(index)}
                className="text-slate-400 hover:text-red-400"
              >
                ×
              </button>
            </div>
          ))}
        </div>
      )}

      <form onSubmit={handleSubmit} className="flex items-end gap-3">
        {/* File upload button */}
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept="image/*,.pdf,.txt,.md,.json,.py,.js,.ts"
          onChange={handleFileSelect}
          className="hidden"
        />
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled}
          className={clsx(
            'p-2 rounded-lg transition-colors',
            disabled
              ? 'text-slate-600 cursor-not-allowed'
              : 'text-slate-400 hover:text-white hover:bg-slate-700'
          )}
        >
          <Paperclip size={20} />
        </button>

        {/* Text input */}
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={content}
            onChange={(e) => setContent(e.target.value)}
            onKeyDown={handleKeyDown}
            onInput={handleInput}
            disabled={disabled}
            placeholder={placeholder}
            rows={1}
            className={clsx(
              'w-full px-4 py-3 bg-slate-700 rounded-xl resize-none',
              'text-white placeholder-slate-400',
              'focus:outline-none focus:ring-2 focus:ring-tok-500',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          />
        </div>

        {/* Send button */}
        <button
          type="submit"
          disabled={disabled || (!content.trim() && files.length === 0)}
          className={clsx(
            'p-3 rounded-xl transition-colors',
            disabled || (!content.trim() && files.length === 0)
              ? 'bg-slate-700 text-slate-500 cursor-not-allowed'
              : 'bg-tok-600 text-white hover:bg-tok-700'
          )}
        >
          {disabled ? (
            <Loader size={20} className="animate-spin" />
          ) : (
            <Send size={20} />
          )}
        </button>
      </form>

      <p className="text-xs text-slate-500 mt-2 text-center">
        Press Enter to send, Shift+Enter for new line
      </p>
    </div>
  );
}
