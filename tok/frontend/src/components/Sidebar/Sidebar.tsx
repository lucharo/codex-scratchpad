import { useState } from 'react';
import { useParams } from 'react-router-dom';
import { TreesPlus, Trees, Trash2, ChevronRight, ChevronDown, GitBranch } from 'lucide-react';
import { deleteTree } from '@/lib/api';
import type { Tree } from '@/types';
import clsx from 'clsx';

interface SidebarProps {
  trees: Tree[];
  loading: boolean;
  onNewTree: () => void;
  onTreeSelect: (tree: Tree) => void;
  onTreeDeleted: (treeId: string) => void;
}

export function Sidebar({
  trees,
  loading,
  onNewTree,
  onTreeSelect,
  onTreeDeleted,
}: SidebarProps) {
  const { treeId } = useParams();
  const [expandedTrees, setExpandedTrees] = useState<Set<string>>(new Set());

  const toggleExpand = (id: string) => {
    setExpandedTrees((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleDelete = async (e: React.MouseEvent, tree: Tree) => {
    e.stopPropagation();
    if (!confirm(`Delete "${tree.name}"? This cannot be undone.`)) return;

    try {
      await deleteTree(tree.id);
      onTreeDeleted(tree.id);
    } catch (err) {
      console.error('Failed to delete tree:', err);
    }
  };

  return (
    <aside className="w-64 bg-slate-800 border-r border-slate-700 flex flex-col">
      {/* Header */}
      <div className="p-4 border-b border-slate-700">
        <div className="flex items-center gap-2 mb-4">
          <span className="text-2xl">🌳</span>
          <h1 className="text-lg font-semibold text-white">Tree of Knowledge</h1>
        </div>
        <button
          onClick={onNewTree}
          className="w-full flex items-center justify-center gap-2 px-4 py-2 bg-tok-600 hover:bg-tok-700 text-white rounded-lg transition-colors"
        >
          <TreesPlus size={18} />
          <span>New Tree</span>
        </button>
      </div>

      {/* Trees List */}
      <div className="flex-1 overflow-y-auto">
        <div className="p-2">
          <h2 className="flex items-center gap-2 px-2 py-1 text-xs font-semibold text-slate-400 uppercase tracking-wider">
            <Trees size={14} />
            All Trees
          </h2>

          {loading ? (
            <div className="px-2 py-4 text-center text-slate-500 text-sm">
              Loading...
            </div>
          ) : trees.length === 0 ? (
            <div className="px-2 py-4 text-center text-slate-500 text-sm">
              No trees yet
            </div>
          ) : (
            <ul className="space-y-1 mt-2">
              {trees.map((tree) => {
                const isActive = tree.id === treeId;
                const isExpanded = expandedTrees.has(tree.id);

                return (
                  <li key={tree.id}>
                    <div
                      className={clsx(
                        'group flex items-center gap-2 px-2 py-2 rounded-lg cursor-pointer transition-colors',
                        isActive
                          ? 'bg-slate-700 text-white'
                          : 'text-slate-300 hover:bg-slate-700/50'
                      )}
                      onClick={() => onTreeSelect(tree)}
                    >
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          toggleExpand(tree.id);
                        }}
                        className="p-0.5 hover:bg-slate-600 rounded"
                      >
                        {isExpanded ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                      </button>

                      <span className="flex-1 truncate text-sm">{tree.name}</span>

                      {tree.branch_count > 1 && (
                        <span className="flex items-center gap-1 text-xs text-slate-500">
                          <GitBranch size={12} />
                          {tree.branch_count}
                        </span>
                      )}

                      <button
                        onClick={(e) => handleDelete(e, tree)}
                        className="p-1 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-400 rounded transition-all"
                      >
                        <Trash2 size={14} />
                      </button>
                    </div>

                    {/* Branch list (placeholder - would need to fetch branches) */}
                    {isExpanded && tree.branch_count > 1 && (
                      <ul className="ml-6 mt-1 space-y-1 border-l border-slate-600 pl-2">
                        <li className="text-xs text-slate-500 py-1">
                          {tree.branch_count} branches
                        </li>
                      </ul>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-700 text-xs text-slate-500">
        <p>Powered by Claude Agent SDK</p>
      </div>
    </aside>
  );
}
