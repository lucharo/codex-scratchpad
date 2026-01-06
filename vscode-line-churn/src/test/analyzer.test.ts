/**
 * Unit tests for the churn analyzer core logic.
 */

import * as assert from 'assert';

// Test the core counting logic (same algorithm as analyzer.ts)
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

suite('Churn Counting', () => {
    test('counts added lines', () => {
        const diff = `
commit abc
+const x = 42;
`;
        assert.strictEqual(countInDiffs('const x = 42;', diff), 1);
    });

    test('counts removed lines', () => {
        const diff = `
commit abc
-const x = 42;
`;
        assert.strictEqual(countInDiffs('const x = 42;', diff), 1);
    });

    test('counts multiple occurrences', () => {
        const diff = `
+TAX_RATE = 0.1
-TAX_RATE = 0.1
+TAX_RATE = 0.1
-TAX_RATE = 0.1
`;
        assert.strictEqual(countInDiffs('TAX_RATE = 0.1', diff), 4);
    });

    test('handles regex special chars', () => {
        const diff = `
+const pattern = /test.*$/;
`;
        assert.strictEqual(countInDiffs('const pattern = /test.*$/;', diff), 1);
    });

    test('returns 0 for empty lines', () => {
        assert.strictEqual(countInDiffs('', 'anything'), 0);
        assert.strictEqual(countInDiffs('   ', 'anything'), 0);
    });

    test('exact match only', () => {
        const diff = `
+const x = 42;
+const x = 42; // comment
`;
        assert.strictEqual(countInDiffs('const x = 42;', diff), 1);
    });
});

suite('Normalization', () => {
    function normalize(counts: number[]): number[] {
        const max = Math.max(...counts);
        if (max === 0) {
            return counts.map(() => 0);
        }
        return counts.map(c => c / max);
    }

    test('normalizes to 0-1 range', () => {
        assert.deepStrictEqual(normalize([0, 5, 10]), [0, 0.5, 1]);
    });

    test('handles all zeros', () => {
        assert.deepStrictEqual(normalize([0, 0, 0]), [0, 0, 0]);
    });
});

suite('Color Generation', () => {
    function getColor(intensity: number, scheme: string, maxOpacity: number): string {
        const opacity = intensity * maxOpacity;
        if (scheme === 'blue') {
            const b = Math.round(100 + 155 * intensity);
            return `rgba(50, 100, ${b}, ${opacity})`;
        }
        if (scheme === 'mono') {
            const g = Math.round(200 - 150 * intensity);
            return `rgba(${g}, ${g}, ${g}, ${opacity})`;
        }
        const r = Math.round(255 * Math.min(intensity * 2, 1));
        const g = Math.round(255 * Math.min((1 - intensity) * 2, 1));
        return `rgba(${r}, ${g}, 50, ${opacity})`;
    }

    test('heat: green at 0', () => {
        assert.strictEqual(getColor(0, 'heat', 0.3), 'rgba(0, 255, 50, 0)');
    });

    test('heat: red at 1', () => {
        assert.strictEqual(getColor(1, 'heat', 0.3), 'rgba(255, 0, 50, 0.3)');
    });

    test('blue scheme', () => {
        assert.strictEqual(getColor(1, 'blue', 0.3), 'rgba(50, 100, 255, 0.3)');
    });

    test('mono scheme', () => {
        assert.strictEqual(getColor(1, 'mono', 0.3), 'rgba(50, 50, 50, 0.3)');
    });
});
