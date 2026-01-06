/**
 * Core type definitions for the Line Churn extension.
 * Centralized types improve maintainability and type safety.
 */

/** Churn data for a single line */
export interface LineChurn {
    /** 0-indexed line number for VS Code compatibility */
    lineNumber: number;
    /** The actual content of the line (trimmed end) */
    content: string;
    /** Raw count of times this line appeared in diffs */
    churnCount: number;
    /** Normalized churn value (0-1) for consistent visualization */
    normalizedChurn: number;
}

/** Complete churn analysis result for a file */
export interface ChurnData {
    /** Absolute path to the analyzed file */
    filePath: string;
    /** Churn data for each line */
    lines: LineChurn[];
    /** Maximum churn count found in the file */
    maxChurn: number;
    /** Timestamp of when analysis was performed */
    analyzedAt: Date;
    /** Git root directory for this file */
    gitRoot: string;
}

/** Supported visualization styles */
export type DisplayStyle = 'gutter' | 'background' | 'both';

/** Supported color schemes */
export type ColorScheme = 'heat' | 'blue' | 'mono';

/** Extension configuration options */
export interface ExtensionConfig {
    enabled: boolean;
    style: DisplayStyle;
    maxOpacity: number;
    colorScheme: ColorScheme;
    commitLimit: number;
    minLineLength: number;
}

/** Result type for operations that can fail */
export type Result<T, E = Error> =
    | { success: true; value: T }
    | { success: false; error: E };

/** Analysis error types for better error handling */
export enum AnalysisErrorType {
    NotGitRepository = 'NOT_GIT_REPOSITORY',
    FileNotTracked = 'FILE_NOT_TRACKED',
    FileNotFound = 'FILE_NOT_FOUND',
    GitCommandFailed = 'GIT_COMMAND_FAILED',
    ParseError = 'PARSE_ERROR',
    Timeout = 'TIMEOUT',
}

export class AnalysisError extends Error {
    constructor(
        public readonly type: AnalysisErrorType,
        message: string,
        public readonly cause?: Error
    ) {
        super(message);
        this.name = 'AnalysisError';
    }
}
