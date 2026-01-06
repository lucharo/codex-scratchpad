import * as vscode from 'vscode';
import { ChurnData, LineChurn } from './churnAnalyzer';

export class DecorationProvider {
    private decorationTypes: vscode.TextEditorDecorationType[] = [];

    /**
     * Apply churn decorations to an editor.
     * Uses bucketed decorations (10 intensity levels) for VS Code efficiency.
     */
    apply(editor: vscode.TextEditor, data: ChurnData) {
        // Clear any existing decorations first
        this.clear(editor);

        const config = vscode.workspace.getConfiguration('lineChurn');
        const style = config.get<string>('style', 'background');
        const maxOpacity = config.get<number>('maxOpacity', 0.3);
        const colorScheme = config.get<string>('colorScheme', 'heat');

        // Group lines by churn buckets for efficiency
        // VS Code performs better with fewer decoration types
        const numBuckets = 10;
        const buckets = this.createBuckets(data.lines, numBuckets);

        for (let bucketIndex = 0; bucketIndex < buckets.length; bucketIndex++) {
            const lines = buckets[bucketIndex];
            if (lines.length === 0) continue;

            const intensity = bucketIndex / (numBuckets - 1);
            const color = this.getColor(intensity, colorScheme, maxOpacity);

            const decorationType = vscode.window.createTextEditorDecorationType({
                backgroundColor: style !== 'gutter' ? color : undefined,
                overviewRulerColor: color,
                overviewRulerLane: vscode.OverviewRulerLane.Right,
                // For gutter mode, we'd generate SVG icons here
                // For MVP, we focus on background highlighting
                isWholeLine: true,
            });

            this.decorationTypes.push(decorationType);

            // Create ranges for all lines in this bucket
            const ranges = lines.map(line =>
                new vscode.Range(
                    line.lineNumber,
                    0,
                    line.lineNumber,
                    line.content.length || 1
                )
            );

            editor.setDecorations(decorationType, ranges);
        }
    }

    /**
     * Group lines into buckets based on normalized churn.
     * Lines with 0 churn are skipped (no decoration).
     */
    private createBuckets(lines: LineChurn[], numBuckets: number): LineChurn[][] {
        const buckets: LineChurn[][] = Array.from({ length: numBuckets }, () => []);

        for (const line of lines) {
            // Skip lines with no churn - they get no decoration
            if (line.churnCount === 0) continue;

            // Map normalized churn to bucket index
            const bucketIndex = Math.min(
                Math.floor(line.normalizedChurn * numBuckets),
                numBuckets - 1
            );
            buckets[bucketIndex].push(line);
        }

        return buckets;
    }

    /**
     * Generate a color based on intensity and color scheme.
     * Returns an rgba() string.
     */
    private getColor(intensity: number, scheme: string, maxOpacity: number): string {
        const opacity = intensity * maxOpacity;

        switch (scheme) {
            case 'heat':
                // Green (stable) -> Yellow -> Red (churny)
                // This is the classic "heat map" visualization
                const r = Math.round(255 * Math.min(intensity * 2, 1));
                const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
                return `rgba(${r}, ${g}, 50, ${opacity})`;

            case 'blue':
                // Light blue -> Dark blue
                // Good for users who prefer a calmer color scheme
                const blue = Math.round(100 + 155 * intensity);
                return `rgba(50, 100, ${blue}, ${opacity})`;

            case 'mono':
            default:
                // Light gray -> Dark gray
                // Suitable for users who want minimal distraction
                const gray = Math.round(200 - 150 * intensity);
                return `rgba(${gray}, ${gray}, ${gray}, ${opacity})`;
        }
    }

    /**
     * Clear all decorations from an editor
     */
    clear(editor: vscode.TextEditor) {
        for (const decoration of this.decorationTypes) {
            editor.setDecorations(decoration, []);
            decoration.dispose();
        }
        this.decorationTypes = [];
    }

    /**
     * Dispose all decoration types when extension deactivates
     */
    dispose() {
        for (const decoration of this.decorationTypes) {
            decoration.dispose();
        }
        this.decorationTypes = [];
    }
}
