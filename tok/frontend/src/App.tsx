import { useState, useEffect, useCallback } from 'react';
import { Routes, Route, useNavigate, useParams } from 'react-router-dom';
import { Sidebar } from '@/components/Sidebar/Sidebar';
import { ChatView } from '@/components/Chat/ChatView';
import { listTrees, createTree, getTree } from '@/lib/api';
import type { Tree, TreeDetail, Branch } from '@/types';

function TreeView() {
  const { treeId, branchId } = useParams();
  const navigate = useNavigate();
  const [tree, setTree] = useState<TreeDetail | null>(null);
  const [currentBranch, setCurrentBranch] = useState<Branch | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!treeId) return;

    const loadTree = async () => {
      setLoading(true);
      try {
        const data = await getTree(treeId);
        setTree(data);

        // Set current branch
        if (branchId) {
          const branch = data.branches.find((b) => b.id === branchId);
          setCurrentBranch(branch || null);
        } else {
          // Default to root branch
          const rootBranch = data.branches.find((b) => b.is_root);
          if (rootBranch) {
            setCurrentBranch(rootBranch);
            navigate(`/tree/${treeId}/branch/${rootBranch.id}`, { replace: true });
          }
        }
      } catch (err) {
        console.error('Failed to load tree:', err);
      } finally {
        setLoading(false);
      }
    };

    loadTree();
  }, [treeId, branchId, navigate]);

  const handleBranchSelect = useCallback(
    (branch: Branch) => {
      setCurrentBranch(branch);
      navigate(`/tree/${treeId}/branch/${branch.id}`);
    },
    [treeId, navigate]
  );

  const handleBranchCreated = useCallback(
    (branch: Branch) => {
      if (tree) {
        setTree({
          ...tree,
          branches: [...tree.branches, branch],
        });
        handleBranchSelect(branch);
      }
    },
    [tree, handleBranchSelect]
  );

  const refreshTree = useCallback(async () => {
    if (!treeId) return;
    const data = await getTree(treeId);
    setTree(data);
  }, [treeId]);

  if (loading) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        Loading...
      </div>
    );
  }

  if (!tree || !currentBranch) {
    return (
      <div className="flex-1 flex items-center justify-center text-slate-400">
        Tree not found
      </div>
    );
  }

  return (
    <ChatView
      tree={tree}
      branch={currentBranch}
      onBranchCreated={handleBranchCreated}
      onRefresh={refreshTree}
    />
  );
}

function EmptyState() {
  return (
    <div className="flex-1 flex flex-col items-center justify-center text-slate-400">
      <div className="text-6xl mb-4">🌳</div>
      <h2 className="text-xl font-semibold mb-2">Welcome to Tree of Knowledge</h2>
      <p className="text-sm">Create a new tree to start a conversation</p>
    </div>
  );
}

export default function App() {
  const [trees, setTrees] = useState<Tree[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  const loadTrees = useCallback(async () => {
    try {
      const data = await listTrees();
      setTrees(data);
    } catch (err) {
      console.error('Failed to load trees:', err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadTrees();
  }, [loadTrees]);

  const handleNewTree = useCallback(async () => {
    try {
      const tree = await createTree({ name: 'New Tree' });
      setTrees((prev) => [tree, ...prev]);
      navigate(`/tree/${tree.id}`);
    } catch (err) {
      console.error('Failed to create tree:', err);
    }
  }, [navigate]);

  const handleTreeSelect = useCallback(
    (tree: Tree) => {
      navigate(`/tree/${tree.id}`);
    },
    [navigate]
  );

  const handleTreeDeleted = useCallback((treeId: string) => {
    setTrees((prev) => prev.filter((t) => t.id !== treeId));
  }, []);

  return (
    <div className="flex h-screen bg-slate-900">
      <Sidebar
        trees={trees}
        loading={loading}
        onNewTree={handleNewTree}
        onTreeSelect={handleTreeSelect}
        onTreeDeleted={handleTreeDeleted}
      />
      <main className="flex-1 flex flex-col">
        <Routes>
          <Route path="/" element={<EmptyState />} />
          <Route path="/tree/:treeId" element={<TreeView />} />
          <Route path="/tree/:treeId/branch/:branchId" element={<TreeView />} />
        </Routes>
      </main>
    </div>
  );
}
