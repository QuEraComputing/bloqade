/**
 * xref.ts — build-time API cross-reference RESOLUTION logic.
 *
 * HERMETIC w.r.t. the generated inventory: `xref.ts` statically imports
 * `../generated/xref-inventory.json`, which is gitignored + build-generated and
 * may be absent (or, in this working tree, a real multi-MB file). We `vi.mock`
 * that JSON module with a small controlled fixture so the tests assert the
 * resolution ALGORITHM (version-preferred -> agnostic fallback -> null) against
 * known data, independent of any prior build. The `@/generated/...` alias and
 * xref.ts's own `../generated/...` specifier resolve to the SAME absolute
 * module, so the mock intercepts it.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest';

// `vi.hoisted` so the fixture is available inside the hoisted `vi.mock` factory.
const { FIXTURE } = vi.hoisted(() => ({
  FIXTURE: {
    byVersion: {
      dev: {
        'bloqade.task.Task': '/api/dev/python/bloqade/task/#bloqade.task.Task',
        'bloqade.shared.Sym': '/api/dev/python/bloqade/shared/#bloqade.shared.Sym',
      },
      '0.35': {
        'bloqade.task.Task': '/api/0.35/python/bloqade/task/#bloqade.task.Task',
      },
    },
    any: {
      'bloqade.shared.Sym': '/api/dev/python/bloqade/shared/#bloqade.shared.Sym',
      'bloqade.only.Agnostic': '/api/dev/python/bloqade/only/#bloqade.only.Agnostic',
    },
  },
}));

vi.mock('@/generated/xref-inventory.json', () => ({ default: FIXTURE }));

// Imported AFTER the mock is registered (vi.mock is hoisted above imports).
import { hasXref, resolveXref } from '@/lib/xref';

describe('resolveXref', () => {
  beforeEach(() => vi.clearAllMocks());

  it('prefers the referring version when the id exists there', () => {
    expect(resolveXref('bloqade.task.Task', '0.35')).toBe(
      '/api/0.35/python/bloqade/task/#bloqade.task.Task',
    );
    expect(resolveXref('bloqade.task.Task', 'dev')).toBe(
      '/api/dev/python/bloqade/task/#bloqade.task.Task',
    );
  });

  it('falls back to the version-agnostic map when the version lacks the id', () => {
    // '0.35' has no 'bloqade.shared.Sym', so it resolves via `any`.
    expect(resolveXref('bloqade.shared.Sym', '0.35')).toBe(
      '/api/dev/python/bloqade/shared/#bloqade.shared.Sym',
    );
  });

  it('uses the agnostic map when no version is supplied', () => {
    expect(resolveXref('bloqade.only.Agnostic')).toBe(
      '/api/dev/python/bloqade/only/#bloqade.only.Agnostic',
    );
  });

  it('falls back to agnostic for an unknown version', () => {
    expect(resolveXref('bloqade.only.Agnostic', 'does-not-exist')).toBe(
      '/api/dev/python/bloqade/only/#bloqade.only.Agnostic',
    );
  });

  it('returns null when the id is in neither the version nor the agnostic map', () => {
    expect(resolveXref('bloqade.nope.Missing', 'dev')).toBeNull();
    expect(resolveXref('bloqade.nope.Missing')).toBeNull();
  });

  it('does NOT cross-contaminate: a version-only id is null with no/other version', () => {
    // 'bloqade.task.Task' lives in byVersion but NOT in `any`.
    expect(resolveXref('bloqade.task.Task')).toBeNull();
  });
});

describe('hasXref', () => {
  it('mirrors resolveXref truthiness', () => {
    expect(hasXref('bloqade.task.Task', 'dev')).toBe(true);
    expect(hasXref('bloqade.only.Agnostic')).toBe(true);
    expect(hasXref('bloqade.nope.Missing')).toBe(false);
    expect(hasXref('bloqade.task.Task')).toBe(false); // version-only id, no version
  });
});
