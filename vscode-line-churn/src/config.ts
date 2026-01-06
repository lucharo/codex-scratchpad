/**
 * Configuration service for centralized access to extension settings.
 * Provides type-safe access with defaults and validation.
 */

import * as vscode from 'vscode';
import { ExtensionConfig, DisplayStyle, ColorScheme } from './types';

const CONFIG_SECTION = 'lineChurn';

/** Default configuration values */
const DEFAULTS: ExtensionConfig = {
    enabled: true,
    style: 'background',
    maxOpacity: 0.3,
    colorScheme: 'heat',
    commitLimit: 500,
    minLineLength: 3,
};

/**
 * Get the current extension configuration with type safety and defaults.
 */
export function getConfig(): ExtensionConfig {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);

    return {
        enabled: config.get<boolean>('enabled', DEFAULTS.enabled),
        style: validateStyle(config.get<string>('style', DEFAULTS.style)),
        maxOpacity: clamp(
            config.get<number>('maxOpacity', DEFAULTS.maxOpacity),
            0.1,
            0.8
        ),
        colorScheme: validateColorScheme(
            config.get<string>('colorScheme', DEFAULTS.colorScheme)
        ),
        commitLimit: clamp(
            config.get<number>('commitLimit', DEFAULTS.commitLimit),
            50,
            5000
        ),
        minLineLength: clamp(
            config.get<number>('minLineLength', DEFAULTS.minLineLength),
            0,
            20
        ),
    };
}

/**
 * Update a configuration value.
 */
export async function setConfig<K extends keyof ExtensionConfig>(
    key: K,
    value: ExtensionConfig[K],
    target: vscode.ConfigurationTarget = vscode.ConfigurationTarget.Global
): Promise<void> {
    const config = vscode.workspace.getConfiguration(CONFIG_SECTION);
    await config.update(key, value, target);
}

/**
 * Check if configuration change affects line churn.
 */
export function affectsLineChurn(event: vscode.ConfigurationChangeEvent): boolean {
    return event.affectsConfiguration(CONFIG_SECTION);
}

function validateStyle(value: string): DisplayStyle {
    const valid: DisplayStyle[] = ['gutter', 'background', 'both'];
    return valid.includes(value as DisplayStyle)
        ? (value as DisplayStyle)
        : DEFAULTS.style;
}

function validateColorScheme(value: string): ColorScheme {
    const valid: ColorScheme[] = ['heat', 'blue', 'mono'];
    return valid.includes(value as ColorScheme)
        ? (value as ColorScheme)
        : DEFAULTS.colorScheme;
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}
