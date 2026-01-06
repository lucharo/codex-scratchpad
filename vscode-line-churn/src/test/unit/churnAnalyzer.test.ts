/**
 * Unit tests for churn analysis functions.
 *
 * Uses Mocha test framework.
 * These tests focus on the pure logic functions without git dependencies.
 */

import * as assert from 'assert';

suite('Churn Analysis Logic', () => {
    /**
     * Simulates the core diff counting logic
     */
    function countLineInDiffs(lineContent: string, diffHistory: string): number {
        if (!lineContent.trim()) {
            return 0;
        }

        const escaped = lineContent.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        const addPattern = new RegExp(`\\n\\+${escaped}(?=\\n|$)`, 'g');
        const removePattern = new RegExp(`\\n-${escaped}(?=\\n|$)`, 'g');

        const addMatches = diffHistory.match(addPattern);
        const removeMatches = diffHistory.match(removePattern);

        return (addMatches?.length ?? 0) + (removeMatches?.length ?? 0);
    }

    suite('countLineInDiffs', () => {
        test('should count lines added in diffs', () => {
            const diffHistory = `
commit abc123
--- a/file.ts
+++ b/file.ts
@@ -1,3 +1,4 @@
 unchanged
+const x = 42;
 more unchanged
`;
            assert.strictEqual(countLineInDiffs('const x = 42;', diffHistory), 1);
        });

        test('should count lines removed in diffs', () => {
            const diffHistory = `
commit abc123
--- a/file.ts
+++ b/file.ts
@@ -1,4 +1,3 @@
 unchanged
-const x = 42;
 more unchanged
`;
            assert.strictEqual(countLineInDiffs('const x = 42;', diffHistory), 1);
        });

        test('should count both additions and removals', () => {
            const diffHistory = `
commit abc123
--- a/file.ts
+++ b/file.ts
@@ -1,3 +1,3 @@
 unchanged
-const x = 42;
+const x = 100;

commit def456
--- a/file.ts
+++ b/file.ts
@@ -1,3 +1,3 @@
 unchanged
-const x = 100;
+const x = 42;
`;
            // "const x = 42;" was removed once and added once = 2
            assert.strictEqual(countLineInDiffs('const x = 42;', diffHistory), 2);
        });

        test('should handle lines with special regex characters', () => {
            const diffHistory = `
commit abc123
--- a/file.ts
+++ b/file.ts
@@ -1,3 +1,4 @@
 unchanged
+const pattern = /test.*$/;
 more unchanged
`;
            assert.strictEqual(countLineInDiffs('const pattern = /test.*$/;', diffHistory), 1);
        });

        test('should return 0 for empty lines', () => {
            const diffHistory = `
commit abc123
+const x = 42;
`;
            assert.strictEqual(countLineInDiffs('', diffHistory), 0);
            assert.strictEqual(countLineInDiffs('   ', diffHistory), 0);
        });

        test('should not count lines that are just prefixed', () => {
            const diffHistory = `
commit abc123
+const x = 42;
+const x = 42; // with comment
`;
            // Only exact match should count
            assert.strictEqual(countLineInDiffs('const x = 42;', diffHistory), 1);
        });

        test('should handle multiple occurrences', () => {
            const diffHistory = `
commit 1
+TAX_RATE = 0.1

commit 2
-TAX_RATE = 0.1
+TAX_RATE = 0.15

commit 3
-TAX_RATE = 0.15
+TAX_RATE = 0.1

commit 4
-TAX_RATE = 0.1
+TAX_RATE = 0.08
`;
            // TAX_RATE = 0.1 appears: +1, -1, +1, -1 = 4 times
            assert.strictEqual(countLineInDiffs('TAX_RATE = 0.1', diffHistory), 4);
        });
    });

    suite('normalization', () => {
        function normalizeChurn(counts: number[]): number[] {
            const max = Math.max(...counts);
            if (max === 0) return counts.map(() => 0);
            return counts.map(c => c / max);
        }

        test('should normalize to 0-1 range', () => {
            const normalized = normalizeChurn([0, 5, 10]);
            assert.deepStrictEqual(normalized, [0, 0.5, 1]);
        });

        test('should handle all zeros', () => {
            const normalized = normalizeChurn([0, 0, 0]);
            assert.deepStrictEqual(normalized, [0, 0, 0]);
        });

        test('should handle single value', () => {
            const normalized = normalizeChurn([5]);
            assert.deepStrictEqual(normalized, [1]);
        });
    });

    suite('bucket assignment', () => {
        function assignBucket(normalizedChurn: number, numBuckets: number): number {
            return Math.min(
                Math.floor(normalizedChurn * numBuckets),
                numBuckets - 1
            );
        }

        test('should assign 0 to bucket 0', () => {
            assert.strictEqual(assignBucket(0, 10), 0);
        });

        test('should assign 1.0 to last bucket', () => {
            assert.strictEqual(assignBucket(1.0, 10), 9);
        });

        test('should assign 0.5 to middle bucket', () => {
            assert.strictEqual(assignBucket(0.5, 10), 5);
        });

        test('should clamp values at boundary', () => {
            // Even if somehow > 1, should clamp to last bucket
            assert.strictEqual(assignBucket(1.1, 10), 9);
        });
    });
});

suite('Regex Escaping', () => {
    function escapeRegex(str: string): string {
        return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    test('should escape dots', () => {
        assert.strictEqual(escapeRegex('a.b'), 'a\\.b');
    });

    test('should escape asterisks', () => {
        assert.strictEqual(escapeRegex('a*b'), 'a\\*b');
    });

    test('should escape brackets', () => {
        assert.strictEqual(escapeRegex('[a]'), '\\[a\\]');
    });

    test('should escape complex patterns', () => {
        const input = 'const regex = /^test.*$/;';
        const escaped = escapeRegex(input);
        // Should be able to create a regex from it
        assert.doesNotThrow(() => new RegExp(escaped));
    });

    test('should handle already escaped backslashes', () => {
        assert.strictEqual(escapeRegex('a\\b'), 'a\\\\b');
    });
});

suite('Color Generation', () => {
    function heatColor(intensity: number, opacity: number): string {
        const r = Math.round(255 * Math.min(intensity * 2, 1));
        const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
        return `rgba(${r}, ${g}, 50, ${opacity.toFixed(3)})`;
    }

    test('should generate green for low intensity', () => {
        const color = heatColor(0, 0.3);
        // At intensity 0: r=0, g=255
        assert.strictEqual(color, 'rgba(0, 255, 50, 0.300)');
    });

    test('should generate red for high intensity', () => {
        const color = heatColor(1, 0.3);
        // At intensity 1: r=255, g=0
        assert.strictEqual(color, 'rgba(255, 0, 50, 0.300)');
    });

    test('should generate yellow for mid intensity', () => {
        const color = heatColor(0.5, 0.3);
        // At intensity 0.5: r=255, g=255
        assert.strictEqual(color, 'rgba(255, 255, 50, 0.150)');
    });
});
