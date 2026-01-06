import { useState, useEffect, useRef, useCallback } from 'react';
import { MessageList } from './MessageList';
import { ChatInput } from '@/components/Input/ChatInput';
import { BranchPrompt } from '@/components/Input/BranchPrompt';
import { listMessages, createBranch } from '@/lib/api';
import { useStreaming } from '@/hooks/useStreaming';
import type { TreeDetail, Branch, Message, TextSelection } from '@/types';
import { GitBranch, ArrowLeft } from 'lucide-react';

interface ChatViewProps {
  tree: TreeDetail;
  branch: Branch;
  onBranchCreated: (branch: Branch) => void;
  onRefresh: () => void;
}

export function ChatView({ tree, branch, onBranchCreated, onRefresh }: ChatViewProps) {
  const [messages, setMessages] = useState<Message[]>([]);
  const [selection, setSelection] = useState<TextSelection | null>(null);
  const [showBranchPrompt, setShowBranchPrompt] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const streaming = useStreaming();

  // Load messages
  useEffect(() => {
    const loadMessages = async () => {
      try {
        const data = await listMessages(tree.id, branch.id);
        setMessages(data);
      } catch (err) {
        console.error('Failed to load messages:', err);
      }
    };
    loadMessages();
  }, [tree.id, branch.id]);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, streaming.text]);

  // Handle message send
  const handleSend = useCallback(
    async (content: string) => {
      // Add optimistic user message
      const userMessage: Message = {
        id: `temp_${Date.now()}`,
        branch_id: branch.id,
        role: 'user',
        content,
        thinking: null,
        position: messages.length,
        model: null,
        tool_calls: [],
        attachments: [],
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, userMessage]);

      // Stream response
      await streaming.sendMessage(tree.id, branch.id, content);

      // Refresh messages after streaming completes
      if (streaming.savedMessageId) {
        const data = await listMessages(tree.id, branch.id);
        setMessages(data);
        streaming.reset();
      }
    },
    [tree.id, branch.id, messages.length, streaming]
  );

  // Handle text selection for branching
  const handleTextSelect = useCallback((sel: TextSelection | null) => {
    setSelection(sel);
    if (sel) {
      setShowBranchPrompt(true);
    }
  }, []);

  // Handle branch creation
  const handleCreateBranch = useCallback(
    async (prompt: string) => {
      if (!selection) return;

      try {
        const newBranch = await createBranch(tree.id, {
          source_message_id: selection.messageId,
          source_branch_id: selection.branchId,
          highlight_start: selection.start,
          highlight_end: selection.end,
          highlighted_text: selection.text,
          user_prompt: prompt,
        });

        setShowBranchPrompt(false);
        setSelection(null);
        onBranchCreated(newBranch);
      } catch (err) {
        console.error('Failed to create branch:', err);
      }
    },
    [tree.id, selection, onBranchCreated]
  );

  return (
    <div className="flex-1 flex flex-col h-full">
      {/* Header */}
      <header className="flex items-center gap-4 px-6 py-4 border-b border-slate-700 bg-slate-800/50">
        <div className="flex-1">
          <h1 className="text-lg font-semibold text-white">{tree.name}</h1>
          <div className="flex items-center gap-2 text-sm text-slate-400">
            <GitBranch size={14} />
            <span>{branch.name}</span>
            {!branch.is_root && branch.origin && (
              <span className="text-xs bg-tok-600/20 text-tok-400 px-2 py-0.5 rounded">
                Branched
              </span>
            )}
          </div>
        </div>

        {tree.branches.length > 1 && (
          <div className="text-sm text-slate-400">
            {tree.branches.length} branches
          </div>
        )}
      </header>

      {/* Branch Origin (if applicable) */}
      {branch.origin && (
        <div className="px-6 py-3 bg-tok-900/20 border-b border-tok-700/30">
          <div className="flex items-start gap-3">
            <ArrowLeft size={16} className="mt-1 text-tok-400" />
            <div>
              <p className="text-xs text-tok-400 mb-1">Branched from highlighted text:</p>
              <blockquote className="text-sm text-slate-300 italic border-l-2 border-tok-500 pl-3">
                "{branch.origin.highlighted_text}"
              </blockquote>
              <p className="text-xs text-slate-500 mt-1">
                Your question: {branch.origin.user_prompt}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto">
        <MessageList
          messages={messages}
          branchId={branch.id}
          streamingText={streaming.isStreaming ? streaming.text : null}
          streamingThinking={streaming.isStreaming ? streaming.thinking : null}
          streamingToolCalls={streaming.isStreaming ? streaming.toolCalls : []}
          onTextSelect={handleTextSelect}
        />
        <div ref={messagesEndRef} />
      </div>

      {/* Branch Prompt Modal */}
      {showBranchPrompt && selection && (
        <BranchPrompt
          selectedText={selection.text}
          onSubmit={handleCreateBranch}
          onCancel={() => {
            setShowBranchPrompt(false);
            setSelection(null);
          }}
        />
      )}

      {/* Input */}
      <ChatInput
        onSend={handleSend}
        disabled={streaming.isStreaming}
        placeholder="Send a message... (Select any text above to branch)"
      />
    </div>
  );
}
