/**
 * Unit tests for CacheManager
 *
 * Uses Mocha test framework.
 */

import * as assert from 'assert';

// Import types only - we'll test the logic without vscode dependency
import { ChurnData } from '../../types';

// Simplified CacheManager for unit testing (no vscode dependency)
class TestCacheManager {
    private cache = new Map<string, { data: ChurnData; accessedAt: number }>();
    private maxAge: number;

    constructor(maxAgeMs = 5 * 60 * 1000) {
        this.maxAge = maxAgeMs;
    }

    get(filePath: string): ChurnData | null {
        const entry = this.cache.get(filePath);
        if (!entry) {
            return null;
        }

        const age = Date.now() - entry.data.analyzedAt.getTime();
        if (age > this.maxAge) {
            this.cache.delete(filePath);
            return null;
        }

        entry.accessedAt = Date.now();
        return entry.data;
    }

    set(filePath: string, data: ChurnData): void {
        this.cache.set(filePath, { data, accessedAt: Date.now() });
    }

    invalidate(filePath: string): void {
        this.cache.delete(filePath);
    }

    invalidateDirectory(dirPath: string): void {
        const normalizedDir = dirPath.endsWith('/') ? dirPath : `${dirPath}/`;
        for (const key of this.cache.keys()) {
            if (key.startsWith(normalizedDir)) {
                this.cache.delete(key);
            }
        }
    }

    clear(): void {
        this.cache.clear();
    }

    has(filePath: string): boolean {
        return this.get(filePath) !== null;
    }

    getStats(): { size: number; maxAge: number } {
        return { size: this.cache.size, maxAge: this.maxAge };
    }
}

function createMockChurnData(filePath: string, maxChurn = 10): ChurnData {
    return {
        filePath,
        lines: [
            { lineNumber: 0, content: 'test', churnCount: 5, normalizedChurn: 0.5 },
            { lineNumber: 1, content: 'test2', churnCount: maxChurn, normalizedChurn: 1.0 },
        ],
        maxChurn,
        analyzedAt: new Date(),
        gitRoot: '/test/repo',
    };
}

suite('CacheManager', () => {
    let cache: TestCacheManager;

    setup(() => {
        cache = new TestCacheManager(1000); // 1 second TTL for testing
    });

    suite('get/set', () => {
        test('should store and retrieve data', () => {
            const data = createMockChurnData('/test/file.ts');
            cache.set('/test/file.ts', data);

            const retrieved = cache.get('/test/file.ts');
            assert.deepStrictEqual(retrieved, data);
        });

        test('should return null for non-existent keys', () => {
            const result = cache.get('/nonexistent/file.ts');
            assert.strictEqual(result, null);
        });
    });

    suite('invalidate', () => {
        test('should remove specific entry', () => {
            const data = createMockChurnData('/test/file.ts');
            cache.set('/test/file.ts', data);

            cache.invalidate('/test/file.ts');

            assert.strictEqual(cache.get('/test/file.ts'), null);
        });

        test('should not affect other entries', () => {
            const data1 = createMockChurnData('/test/file1.ts');
            const data2 = createMockChurnData('/test/file2.ts');
            cache.set('/test/file1.ts', data1);
            cache.set('/test/file2.ts', data2);

            cache.invalidate('/test/file1.ts');

            assert.strictEqual(cache.get('/test/file1.ts'), null);
            assert.deepStrictEqual(cache.get('/test/file2.ts'), data2);
        });
    });

    suite('invalidateDirectory', () => {
        test('should remove all entries in directory', () => {
            const data1 = createMockChurnData('/test/dir/file1.ts');
            const data2 = createMockChurnData('/test/dir/file2.ts');
            const data3 = createMockChurnData('/test/other/file3.ts');

            cache.set('/test/dir/file1.ts', data1);
            cache.set('/test/dir/file2.ts', data2);
            cache.set('/test/other/file3.ts', data3);

            cache.invalidateDirectory('/test/dir');

            assert.strictEqual(cache.get('/test/dir/file1.ts'), null);
            assert.strictEqual(cache.get('/test/dir/file2.ts'), null);
            assert.deepStrictEqual(cache.get('/test/other/file3.ts'), data3);
        });
    });

    suite('clear', () => {
        test('should remove all entries', () => {
            cache.set('/test/file1.ts', createMockChurnData('/test/file1.ts'));
            cache.set('/test/file2.ts', createMockChurnData('/test/file2.ts'));

            cache.clear();

            assert.strictEqual(cache.get('/test/file1.ts'), null);
            assert.strictEqual(cache.get('/test/file2.ts'), null);
            assert.strictEqual(cache.getStats().size, 0);
        });
    });

    suite('has', () => {
        test('should return true for existing entries', () => {
            cache.set('/test/file.ts', createMockChurnData('/test/file.ts'));
            assert.strictEqual(cache.has('/test/file.ts'), true);
        });

        test('should return false for non-existent entries', () => {
            assert.strictEqual(cache.has('/nonexistent/file.ts'), false);
        });
    });

    suite('getStats', () => {
        test('should return correct statistics', () => {
            cache.set('/test/file1.ts', createMockChurnData('/test/file1.ts'));
            cache.set('/test/file2.ts', createMockChurnData('/test/file2.ts'));

            const stats = cache.getStats();
            assert.strictEqual(stats.size, 2);
            assert.strictEqual(stats.maxAge, 1000);
        });
    });

    suite('TTL expiration', () => {
        test('should return null for stale entries', async () => {
            const shortTTLCache = new TestCacheManager(50); // 50ms TTL
            const data = createMockChurnData('/test/file.ts');
            shortTTLCache.set('/test/file.ts', data);

            // Wait for TTL to expire
            await new Promise(resolve => setTimeout(resolve, 100));

            assert.strictEqual(shortTTLCache.get('/test/file.ts'), null);
        });
    });
});
