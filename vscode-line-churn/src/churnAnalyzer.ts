/**
 * Git-based line churn analyzer.
 *
 * Core algorithm: Track churn by CONTENT, not line number.
 * This means when code moves (refactoring, adding functions above),
 * the churn history follows the content, not the position.
 */

import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';
import {
    ChurnData,
    LineChurn,
    Result,
    AnalysisError,
    AnalysisErrorType,
} from './types';
import { getConfig } from './config';

const execAsync = promisify(exec);

/** Maximum buffer size for git output (50MB) */
const MAX_BUFFER_SIZE = 50 * 1024 * 1024;

/** Timeout for git commands (30 seconds) */
const GIT_TIMEOUT_MS = 30000;

/**
 * Analyze a file's line-by-line churn using git history.
 *
 * @param filePath - Absolute path to the file to analyze
 * @returns Result containing ChurnData on success, AnalysisError on failure
 */
export async function analyzeFile(
    filePath: string
): Promise<Result<ChurnData, AnalysisError>> {
    try {
        // Validate file exists
        if (!fs.existsSync(filePath)) {
            return {
                success: false,
                error: new AnalysisError(
                    AnalysisErrorType.FileNotFound,
                    `File not found: ${filePath}`
                ),
            };
        }

        // Find git root
        const gitRootResult = await findGitRoot(path.dirname(filePath));
        if (!gitRootResult.success) {
            return gitRootResult;
        }
        const gitRoot = gitRootResult.value;

        // Get diff history
        const relativePath = path.relative(gitRoot, filePath);
        const diffResult = await getGitDiffHistory(gitRoot, relativePath);
        if (!diffResult.success) {
            return diffResult;
        }
        const diffHistory = diffResult.value;

        // Read and analyze current file
        const content = await fs.promises.readFile(filePath, 'utf-8');
        const lines = analyzeLines(content, diffHistory);

        return {
            success: true,
            value: {
                filePath,
                lines: lines.data,
                maxChurn: lines.maxChurn,
                analyzedAt: new Date(),
                gitRoot,
            },
        };
    } catch (error) {
        return {
            success: false,
            error: new AnalysisError(
                AnalysisErrorType.ParseError,
                `Failed to analyze file: ${error instanceof Error ? error.message : String(error)}`,
                error instanceof Error ? error : undefined
            ),
        };
    }
}

/**
 * Find the git repository root by walking up the directory tree.
 */
async function findGitRoot(
    startDir: string
): Promise<Result<string, AnalysisError>> {
    let dir = startDir;
    const root = path.parse(dir).root;

    while (dir !== root) {
        const gitDir = path.join(dir, '.git');
        try {
            const stat = await fs.promises.stat(gitDir);
            if (stat.isDirectory() || stat.isFile()) {
                // .git can be a file in worktrees
                return { success: true, value: dir };
            }
        } catch {
            // .git doesn't exist at this level, continue up
        }
        dir = path.dirname(dir);
    }

    return {
        success: false,
        error: new AnalysisError(
            AnalysisErrorType.NotGitRepository,
            'Not inside a git repository'
        ),
    };
}

/**
 * Get the full diff history for a file using git log -p.
 * Uses --follow to track renames.
 */
async function getGitDiffHistory(
    gitRoot: string,
    relativePath: string
): Promise<Result<string, AnalysisError>> {
    const config = getConfig();

    try {
        const { stdout } = await execAsync(
            `git log -${config.commitLimit} -p --follow -- "${escapePath(relativePath)}"`,
            {
                cwd: gitRoot,
                maxBuffer: MAX_BUFFER_SIZE,
                timeout: GIT_TIMEOUT_MS,
            }
        );
        return { success: true, value: stdout };
    } catch (error) {
        // Check if it's a timeout
        if (error instanceof Error && error.message.includes('TIMEOUT')) {
            return {
                success: false,
                error: new AnalysisError(
                    AnalysisErrorType.Timeout,
                    'Git command timed out - try reducing commit limit',
                    error
                ),
            };
        }

        // File might not be tracked
        return {
            success: false,
            error: new AnalysisError(
                AnalysisErrorType.FileNotTracked,
                'File is not tracked by git or has no history',
                error instanceof Error ? error : undefined
            ),
        };
    }
}

/**
 * Analyze each line's churn count from the diff history.
 */
function analyzeLines(
    content: string,
    diffHistory: string
): { data: LineChurn[]; maxChurn: number } {
    const config = getConfig();
    const currentLines = content.split('\n');
    const lines: LineChurn[] = [];
    let maxChurn = 0;

    for (let i = 0; i < currentLines.length; i++) {
        const lineContent = currentLines[i].trimEnd();
        const trimmedContent = lineContent.trim();
        let churnCount = 0;

        // Skip empty lines and lines shorter than configured minimum
        // Short lines like "}", "pass", "return" tend to have inflated counts
        if (trimmedContent.length >= config.minLineLength) {
            churnCount = countLineInDiffs(lineContent, diffHistory);
        }

        maxChurn = Math.max(maxChurn, churnCount);
        lines.push({
            lineNumber: i,
            content: lineContent,
            churnCount,
            normalizedChurn: 0, // Calculated after we know max
        });
    }

    // Normalize churn values to 0-1 scale for consistent visualization
    if (maxChurn > 0) {
        for (const line of lines) {
            line.normalizedChurn = line.churnCount / maxChurn;
        }
    }

    return { data: lines, maxChurn };
}

/**
 * Count how many times a line appears in the diff history.
 * Lines in diffs are prefixed with + (added) or - (removed).
 */
function countLineInDiffs(lineContent: string, diffHistory: string): number {
    if (!lineContent.trim()) {
        return 0;
    }

    // Escape special regex characters for accurate matching
    const escaped = escapeRegex(lineContent);

    // Count lines added (+) and removed (-) in diffs
    // Use lookahead to handle lines at end of diff blocks
    const addPattern = new RegExp(`\\n\\+${escaped}(?=\\n|$)`, 'g');
    const removePattern = new RegExp(`\\n-${escaped}(?=\\n|$)`, 'g');

    const addMatches = diffHistory.match(addPattern);
    const removeMatches = diffHistory.match(removePattern);

    return (addMatches?.length ?? 0) + (removeMatches?.length ?? 0);
}

/**
 * Escape special regex characters in a string.
 */
function escapeRegex(str: string): string {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/**
 * Escape path for shell command (handle spaces, special chars).
 */
function escapePath(filePath: string): string {
    // Already quoted in the command, just escape internal quotes
    return filePath.replace(/"/g, '\\"');
}
