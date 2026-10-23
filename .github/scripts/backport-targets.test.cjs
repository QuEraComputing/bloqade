const assert = require('node:assert/strict');
const { test } = require('node:test');
const { targetBranches, prepare } = require('./backport-targets.cjs');

const labels = (...names) => names.map(name => ({ name }));

test('maps Lanes-style labels to release branches, with duplicates removed', () => {
  assert.deepEqual(targetBranches(labels('backport v0.35', 'bug', 'backport v0.36', 'backport v0.35')),
    ['release-0-35', 'release-0-36']);
});

test('ignores malformed labels and arbitrary refs or shell expressions', () => {
  assert.deepEqual(targetBranches(labels('backport main', 'backport v0.36.1',
    'backport v0x36', 'backport v0.36; echo bad', 'backport v0.36\n',
    'backport v0.36/other', 'backport v$(id).1')), []);
});

function fixture(overrides = {}) {
  const pr = { number: 42, merged: true,
    base: { ref: 'main', repo: { full_name: 'QuEraComputing/bloqade' } },
    labels: labels('backport v0.36'), ...overrides };
  const outputs = {};
  const branches = [];
  const github = { rest: {
    pulls: { get: async ({ pull_number }) => {
      assert.equal(pull_number, 42);
      return { data: pr };
    } },
    repos: { getBranch: async ({ branch }) => { branches.push(branch); } },
  } };
  return { github, context: { repo: { owner: 'QuEraComputing', repo: 'bloqade' } },
    core: { setOutput: (key, value) => { outputs[key] = value; }, info: () => {} },
    pullNumber: '42', outputs, branches };
}

test('checks existing targets and emits action inputs from current PR labels', async () => {
  const f = fixture({ labels: labels('backport v0.35', 'backport v0.36') });
  await prepare(f);
  assert.deepEqual(f.branches, ['release-0-35', 'release-0-36']);
  assert.deepEqual(f.outputs, { pull_number: '42', targets: 'release-0-35 release-0-36' });
});

test('unlabeled merged PR is a no-op', async () => {
  const f = fixture({ labels: labels('bug') });
  await prepare(f);
  assert.equal(f.outputs.targets, '');
  assert.deepEqual(f.branches, []);
});

test('rejects invalid manual PR numbers before any API request', async () => {
  for (const pullNumber of ['', '0', '-1', '1.2', '42; echo bad', '42\n']) {
    const f = fixture();
    await assert.rejects(prepare({ ...f, pullNumber }), /positive integer/);
  }
});

test('rejects unmerged PRs, release-branch PRs, and a different repository', async () => {
  for (const overrides of [
    { merged: false },
    { base: { ref: 'release-0-36', repo: { full_name: 'QuEraComputing/bloqade' } } },
    { base: { ref: 'main', repo: { full_name: 'someone/fork' } } },
  ]) {
    const f = fixture(overrides);
    await assert.rejects(prepare(f), /Only PRs merged/);
    assert.deepEqual(f.branches, []);
  }
});

test('missing target branches fail before any backport is started', async () => {
  const f = fixture();
  f.github.rest.repos.getBranch = async () => { throw { status: 404 }; };
  await assert.rejects(prepare(f), /Target release-0-36 does not exist/);
  assert.deepEqual(f.outputs, {});
});

test('permission and network errors are not mistaken for missing branches', async () => {
  const f = fixture();
  const failure = new Error('API unavailable');
  f.github.rest.repos.getBranch = async () => { throw failure; };
  await assert.rejects(prepare(f), error => error === failure);
});
