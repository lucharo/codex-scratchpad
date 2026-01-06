import { useState, useCallback, useEffect } from 'react';
import type { TextSelection } from '@/types';

interface UseTextSelectionOptions {
  messageId: string;
  branchId: string;
  containerRef: React.RefObject<HTMLElement>;
}

export function useTextSelection({ messageId, branchId, containerRef }: UseTextSelectionOptions) {
  const [selection, setSelection] = useState<TextSelection | null>(null);

  const handleSelectionChange = useCallback(() => {
    const sel = window.getSelection();
    if (!sel || sel.isCollapsed || !containerRef.current) {
      setSelection(null);
      return;
    }

    // Check if selection is within our container
    const range = sel.getRangeAt(0);
    if (!containerRef.current.contains(range.commonAncestorContainer)) {
      setSelection(null);
      return;
    }

    const text = sel.toString().trim();
    if (!text) {
      setSelection(null);
      return;
    }

    // Calculate offset within the message content
    const preSelectionRange = range.cloneRange();
    preSelectionRange.selectNodeContents(containerRef.current);
    preSelectionRange.setEnd(range.startContainer, range.startOffset);
    const start = preSelectionRange.toString().length;
    const end = start + text.length;

    setSelection({
      messageId,
      branchId,
      start,
      end,
      text,
    });
  }, [messageId, branchId, containerRef]);

  useEffect(() => {
    document.addEventListener('selectionchange', handleSelectionChange);
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange);
    };
  }, [handleSelectionChange]);

  const clearSelection = useCallback(() => {
    setSelection(null);
    window.getSelection()?.removeAllRanges();
  }, []);

  return { selection, clearSelection };
}
