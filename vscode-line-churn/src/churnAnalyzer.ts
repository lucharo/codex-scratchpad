import * as vscode from 'vscode';
import { execSync } from 'child_process';
import * as path from 'path';
import * as fs from 'fs';

export interface LineChurn {
    lineNumber: number;  // 0-indexed for VS Code
    content: string;
    churnCount: number;
    normalizedChurn: number;  // 0-1 scale for consistent coloring
}

export interface ChurnData {
    filePath: string;
    lines: LineChurn[];
    maxChurn: number;
    analyzedAt: Date;
}

export class ChurnAnalyzer {

    /**
     * Analyze a file's line-by-line churn using git history.
     * The key insight: track by CONTENT, not line number.
     * This means churn history follows code when it moves around.
     */
    async analyze(filePath: string): Promise<ChurnData | null> {
        try {
            const config = vscode.workspace.getConfiguration('lineChurn');
            const commitLimit = config.get<number>('commitLimit', 500);
            const minLineLength = config.get<number>('minLineLength', 3);

            // Find git root directory
            const dir = path.dirname(filePath);
            const gitRoot = this.findGitRoot(dir);
            if (!gitRoot) {
                return null;
            }

            // Get diff history for this file
            const relativePath = path.relative(gitRoot, filePath);
            const diffHistory = this.getGitDiffHistory(gitRoot, relativePath, commitLimit);
            if (!diffHistory) {
                return null;
            }

            // Read current file content
            const content = fs.readFileSync(filePath, 'utf-8');
            const currentLines = content.split('\n');

            // Analyze each line
            const lines: LineChurn[] = [];
            let maxChurn = 0;

            for (let i = 0; i < currentLines.length; i++) {
                const lineContent = currentLines[i].trimEnd();
                const trimmedContent = lineContent.trim();
                let churnCount = 0;

                // Skip empty lines and very short lines (configurable)
                // Short lines like "pass", "}", "return" tend to have inflated counts
                if (trimmedContent.length >= minLineLength) {
                    churnCount = this.countLineInDiffs(lineContent, diffHistory);
                }

                maxChurn = Math.max(maxChurn, churnCount);
                lines.push({
                    lineNumber: i,
                    content: lineContent,
                    churnCount,
                    normalizedChurn: 0  // Will normalize after we know max
                });
            }

            // Normalize churn values to 0-1 scale
            // This ensures consistent coloring across files with different churn ranges
            if (maxChurn > 0) {
                for (const line of lines) {
                    line.normalizedChurn = line.churnCount / maxChurn;
                }
            }

            return {
                filePath,
                lines,
                maxChurn,
                analyzedAt: new Date()
            };

        } catch (error) {
            console.error('ChurnAnalyzer error:', error);
            return null;
        }
    }

    /**
     * Find the git repository root by walking up the directory tree
     */
    private findGitRoot(startDir: string): string | null {
        let dir = startDir;
        while (dir !== path.dirname(dir)) {
            if (fs.existsSync(path.join(dir, '.git'))) {
                return dir;
            }
            dir = path.dirname(dir);
        }
        return null;
    }

    /**
     * Get the full diff history for a file using git log -p
     * The --follow flag tracks renames
     */
    private getGitDiffHistory(
        gitRoot: string,
        relativePath: string,
        commitLimit: number
    ): string | null {
        try {
            // Use -p to get patches, --follow to track renames
            // Limit commits for performance in large repos
            const result = execSync(
                `git log -${commitLimit} -p --follow -- "${relativePath}"`,
                {
                    cwd: gitRoot,
                    maxBuffer: 50 * 1024 * 1024,  // 50MB buffer for large histories
                    encoding: 'utf-8',
                    stdio: ['pipe', 'pipe', 'pipe']  // Suppress stderr
                }
            );
            return result;
        } catch (error) {
            // File might not be tracked by git
            return null;
        }
    }

    /**
     * Count how many times a line appears in the diff history.
     * Lines in diffs are prefixed with + (added) or - (removed).
     * A line that changes shows up as both removed and added.
     */
    private countLineInDiffs(lineContent: string, diffHistory: string): number {
        if (!lineContent.trim()) {
            return 0;
        }

        // Escape special regex characters for accurate matching
        const escaped = this.escapeForRegex(lineContent);

        // Count lines added (+) and removed (-) in diffs
        // Use lookahead to handle lines at end of diff blocks
        const addPattern = new RegExp(`\\n\\+${escaped}(?=\\n|$)`, 'g');
        const removePattern = new RegExp(`\\n-${escaped}(?=\\n|$)`, 'g');

        const addMatches = diffHistory.match(addPattern);
        const removeMatches = diffHistory.match(removePattern);

        return (addMatches?.length || 0) + (removeMatches?.length || 0);
    }

    /**
     * Escape special regex characters in a string
     */
    private escapeForRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }
}
