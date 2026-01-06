/**
 * Decoration provider for visualizing line churn in VS Code.
 *
 * Uses bucketed decorations (10 intensity levels) for VS Code efficiency.
 * Creates fewer TextEditorDecorationType instances which improves performance.
 */

import * as vscode from 'vscode';
import { ChurnData, LineChurn, ColorScheme, DisplayStyle } from './types';
import { getConfig } from './config';

/** Number of intensity buckets for decorations */
const NUM_BUCKETS = 10;

/**
 * Manages text editor decorations for churn visualization.
 * Implements Disposable for proper cleanup.
 */
export class DecorationProvider implements vscode.Disposable {
    private decorationTypes: vscode.TextEditorDecorationType[] = [];
    private currentEditor: vscode.TextEditor | null = null;

    /**
     * Apply churn decorations to an editor.
     */
    apply(editor: vscode.TextEditor, data: ChurnData): void {
        // Clear existing decorations first
        this.clearDecorations();

        const config = getConfig();
        const { style, maxOpacity, colorScheme } = config;

        // Group lines by churn buckets
        const buckets = this.createBuckets(data.lines);

        // Create decoration types for each non-empty bucket
        for (let bucketIndex = 0; bucketIndex < buckets.length; bucketIndex++) {
            const lines = buckets[bucketIndex];
            if (lines.length === 0) {
                continue;
            }

            const intensity = bucketIndex / (NUM_BUCKETS - 1);
            const decorationType = this.createDecorationType(
                intensity,
                style,
                colorScheme,
                maxOpacity
            );

            this.decorationTypes.push(decorationType);

            // Create ranges for all lines in this bucket
            const ranges = lines.map((line) =>
                new vscode.Range(
                    line.lineNumber,
                    0,
                    line.lineNumber,
                    Math.max(line.content.length, 1)
                )
            );

            editor.setDecorations(decorationType, ranges);
        }

        this.currentEditor = editor;
    }

    /**
     * Clear all decorations from a specific editor.
     */
    clear(editor: vscode.TextEditor): void {
        for (const decoration of this.decorationTypes) {
            editor.setDecorations(decoration, []);
        }
        this.clearDecorations();
    }

    /**
     * Create decoration type for a given intensity level.
     */
    private createDecorationType(
        intensity: number,
        style: DisplayStyle,
        colorScheme: ColorScheme,
        maxOpacity: number
    ): vscode.TextEditorDecorationType {
        const color = this.getColor(intensity, colorScheme, maxOpacity);

        const options: vscode.DecorationRenderOptions = {
            isWholeLine: true,
            overviewRulerColor: color,
            overviewRulerLane: vscode.OverviewRulerLane.Right,
        };

        // Apply background color if style includes it
        if (style === 'background' || style === 'both') {
            options.backgroundColor = color;
        }

        // Apply gutter decoration if style includes it
        if (style === 'gutter' || style === 'both') {
            options.gutterIconPath = this.createGutterIcon(intensity, colorScheme);
            options.gutterIconSize = 'contain';
        }

        return vscode.window.createTextEditorDecorationType(options);
    }

    /**
     * Group lines into buckets based on normalized churn.
     * Lines with 0 churn are skipped (no decoration needed).
     */
    private createBuckets(lines: LineChurn[]): LineChurn[][] {
        const buckets: LineChurn[][] = Array.from(
            { length: NUM_BUCKETS },
            () => []
        );

        for (const line of lines) {
            // Skip lines with no churn - they get no decoration
            if (line.churnCount === 0) {
                continue;
            }

            // Map normalized churn to bucket index
            const bucketIndex = Math.min(
                Math.floor(line.normalizedChurn * NUM_BUCKETS),
                NUM_BUCKETS - 1
            );
            buckets[bucketIndex].push(line);
        }

        return buckets;
    }

    /**
     * Generate an RGBA color string based on intensity and scheme.
     */
    private getColor(
        intensity: number,
        scheme: ColorScheme,
        maxOpacity: number
    ): string {
        const opacity = intensity * maxOpacity;

        switch (scheme) {
            case 'heat':
                // Green (stable) -> Yellow -> Red (churny)
                return this.heatColor(intensity, opacity);

            case 'blue':
                // Light blue -> Dark blue
                return this.blueColor(intensity, opacity);

            case 'mono':
            default:
                // Light gray -> Dark gray
                return this.monoColor(intensity, opacity);
        }
    }

    private heatColor(intensity: number, opacity: number): string {
        // Interpolate: green (0) -> yellow (0.5) -> red (1)
        const r = Math.round(255 * Math.min(intensity * 2, 1));
        const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
        return `rgba(${r}, ${g}, 50, ${opacity.toFixed(3)})`;
    }

    private blueColor(intensity: number, opacity: number): string {
        const blue = Math.round(100 + 155 * intensity);
        return `rgba(50, 100, ${blue}, ${opacity.toFixed(3)})`;
    }

    private monoColor(intensity: number, opacity: number): string {
        const gray = Math.round(200 - 150 * intensity);
        return `rgba(${gray}, ${gray}, ${gray}, ${opacity.toFixed(3)})`;
    }

    /**
     * Create a data URI for a gutter icon SVG.
     */
    private createGutterIcon(
        intensity: number,
        scheme: ColorScheme
    ): vscode.Uri {
        // Get base color without opacity for the icon
        const color = this.getIconColor(intensity, scheme);

        const svg = `
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16">
                <circle cx="8" cy="8" r="${3 + intensity * 3}" fill="${color}" opacity="${0.5 + intensity * 0.5}"/>
            </svg>
        `.trim();

        const encoded = Buffer.from(svg).toString('base64');
        return vscode.Uri.parse(`data:image/svg+xml;base64,${encoded}`);
    }

    private getIconColor(intensity: number, scheme: ColorScheme): string {
        switch (scheme) {
            case 'heat': {
                const r = Math.round(255 * Math.min(intensity * 2, 1));
                const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
                return `rgb(${r}, ${g}, 50)`;
            }
            case 'blue': {
                const blue = Math.round(100 + 155 * intensity);
                return `rgb(50, 100, ${blue})`;
            }
            case 'mono':
            default: {
                const gray = Math.round(200 - 150 * intensity);
                return `rgb(${gray}, ${gray}, ${gray})`;
            }
        }
    }

    /**
     * Clear and dispose all decoration types.
     */
    private clearDecorations(): void {
        for (const decoration of this.decorationTypes) {
            decoration.dispose();
        }
        this.decorationTypes = [];
    }

    /**
     * Dispose all resources.
     */
    dispose(): void {
        this.clearDecorations();
        this.currentEditor = null;
    }
}
