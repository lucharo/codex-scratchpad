/**
 * Line churn analyzer - core algorithm.
 * Tracks churn by CONTENT, not line number, so history follows code when it moves.
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';

const execAsync = promisify(exec);

export interface LineChurn {
    line: number;
    content: string;
    count: number;
    normalized: number;
}

export interface ChurnResult {
    lines: LineChurn[];
    maxChurn: number;
}

/**
 * Analyze line churn for a file using git history.
 */
export async function analyzeFile(filePath: string): Promise<ChurnResult | null> {
    const gitRoot = findGitRoot(path.dirname(filePath));
    if (gitRoot === null) {
        return null;
    }

    const diffHistory = await getGitHistory(gitRoot, filePath);
    if (diffHistory === null) {
        return null;
    }

    const content = fs.readFileSync(filePath, 'utf-8');
    return analyzeLines(content, diffHistory);
}

function findGitRoot(dir: string): string | null {
    let current = dir;
    while (current !== path.dirname(current)) {
        if (fs.existsSync(path.join(current, '.git'))) {
            return current;
        }
        current = path.dirname(current);
    }
    return null;
}

async function getGitHistory(gitRoot: string, filePath: string): Promise<string | null> {
    const rel = path.relative(gitRoot, filePath);
    try {
        const { stdout } = await execAsync(
            `git log -500 -p --follow -- "${rel}"`,
            { cwd: gitRoot, maxBuffer: 50 * 1024 * 1024 }
        );
        return stdout;
    } catch {
        return null;
    }
}

function analyzeLines(content: string, diffHistory: string): ChurnResult {
    const lines = content.split('\n');
    const result: LineChurn[] = [];
    let maxChurn = 0;

    for (let i = 0; i < lines.length; i++) {
        const lineContent = lines[i].trimEnd();
        const trimmed = lineContent.trim();

        let count = 0;
        if (trimmed.length >= 3) {
            count = countInDiffs(lineContent, diffHistory);
        }

        maxChurn = Math.max(maxChurn, count);
        result.push({ line: i, content: lineContent, count, normalized: 0 });
    }

    // Normalize
    if (maxChurn > 0) {
        for (const line of result) {
            line.normalized = line.count / maxChurn;
        }
    }

    return { lines: result, maxChurn };
}

function countInDiffs(lineContent: string, diffHistory: string): number {
    if (!lineContent.trim()) {
        return 0;
    }
    const escaped = lineContent.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const addPattern = new RegExp(`\\n\\+${escaped}(?=\\n|$)`, 'g');
    const removePattern = new RegExp(`\\n-${escaped}(?=\\n|$)`, 'g');

    const adds = diffHistory.match(addPattern)?.length ?? 0;
    const removes = diffHistory.match(removePattern)?.length ?? 0;
    return adds + removes;
}
