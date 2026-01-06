/**
 * Unit tests for config validation logic.
 *
 * Tests pure validation functions without VS Code dependency.
 */

import * as assert from 'assert';
import { DisplayStyle, ColorScheme } from '../../types';

// Re-implement the pure validation functions for testing
// (These match the implementations in config.ts)

function validateStyle(value: string): DisplayStyle {
    const valid: DisplayStyle[] = ['gutter', 'background', 'both'];
    return valid.includes(value as DisplayStyle)
        ? (value as DisplayStyle)
        : 'background';  // default
}

function validateColorScheme(value: string): ColorScheme {
    const valid: ColorScheme[] = ['heat', 'blue', 'mono'];
    return valid.includes(value as ColorScheme)
        ? (value as ColorScheme)
        : 'heat';  // default
}

function clamp(value: number, min: number, max: number): number {
    return Math.min(Math.max(value, min), max);
}

suite('Config Validation', () => {
    suite('validateStyle', () => {
        test('should accept valid style "gutter"', () => {
            assert.strictEqual(validateStyle('gutter'), 'gutter');
        });

        test('should accept valid style "background"', () => {
            assert.strictEqual(validateStyle('background'), 'background');
        });

        test('should accept valid style "both"', () => {
            assert.strictEqual(validateStyle('both'), 'both');
        });

        test('should return default for invalid style', () => {
            assert.strictEqual(validateStyle('invalid'), 'background');
        });

        test('should return default for empty string', () => {
            assert.strictEqual(validateStyle(''), 'background');
        });

        test('should be case-sensitive', () => {
            assert.strictEqual(validateStyle('GUTTER'), 'background');
            assert.strictEqual(validateStyle('Background'), 'background');
        });
    });

    suite('validateColorScheme', () => {
        test('should accept valid scheme "heat"', () => {
            assert.strictEqual(validateColorScheme('heat'), 'heat');
        });

        test('should accept valid scheme "blue"', () => {
            assert.strictEqual(validateColorScheme('blue'), 'blue');
        });

        test('should accept valid scheme "mono"', () => {
            assert.strictEqual(validateColorScheme('mono'), 'mono');
        });

        test('should return default for invalid scheme', () => {
            assert.strictEqual(validateColorScheme('rainbow'), 'heat');
        });

        test('should return default for empty string', () => {
            assert.strictEqual(validateColorScheme(''), 'heat');
        });
    });

    suite('clamp', () => {
        test('should return value when within range', () => {
            assert.strictEqual(clamp(5, 0, 10), 5);
            assert.strictEqual(clamp(0.5, 0.1, 0.8), 0.5);
        });

        test('should clamp to min when below range', () => {
            assert.strictEqual(clamp(-5, 0, 10), 0);
            assert.strictEqual(clamp(0.05, 0.1, 0.8), 0.1);
        });

        test('should clamp to max when above range', () => {
            assert.strictEqual(clamp(15, 0, 10), 10);
            assert.strictEqual(clamp(0.9, 0.1, 0.8), 0.8);
        });

        test('should handle edge cases at boundaries', () => {
            assert.strictEqual(clamp(0, 0, 10), 0);
            assert.strictEqual(clamp(10, 0, 10), 10);
        });

        test('should handle negative ranges', () => {
            assert.strictEqual(clamp(-5, -10, -1), -5);
            assert.strictEqual(clamp(0, -10, -1), -1);
            assert.strictEqual(clamp(-15, -10, -1), -10);
        });

        // Config-specific clamp tests
        test('should clamp maxOpacity to valid range (0.1-0.8)', () => {
            assert.strictEqual(clamp(0.3, 0.1, 0.8), 0.3);
            assert.strictEqual(clamp(0.0, 0.1, 0.8), 0.1);
            assert.strictEqual(clamp(1.0, 0.1, 0.8), 0.8);
        });

        test('should clamp commitLimit to valid range (50-5000)', () => {
            assert.strictEqual(clamp(500, 50, 5000), 500);
            assert.strictEqual(clamp(10, 50, 5000), 50);
            assert.strictEqual(clamp(10000, 50, 5000), 5000);
        });

        test('should clamp minLineLength to valid range (0-20)', () => {
            assert.strictEqual(clamp(3, 0, 20), 3);
            assert.strictEqual(clamp(-1, 0, 20), 0);
            assert.strictEqual(clamp(50, 0, 20), 20);
        });
    });
});
