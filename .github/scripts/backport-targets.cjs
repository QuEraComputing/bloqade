// Loaded only from main, before the backport action changes the checkout.
function targetBranches(labels) {
  return [...new Set(labels.flatMap(({ name }) => {
    const match = /^backport v(\d+)\.(\d+)$/.exec(name);
    return match ? [`release-${match[1]}-${match[2]}`] : [];
  }))];
}

async function prepare({ github, context, core, pullNumber }) {
  if (!/^[1-9]\d*$/.test(String(pullNumber))) {
    throw new Error('Provide a positive integer source PR number.');
  }
  const { data: pr } = await github.rest.pulls.get({
    ...context.repo,
    pull_number: Number(pullNumber),
  });
  if (!pr.merged || pr.base.ref !== 'main' ||
      pr.base.repo.full_name !== `${context.repo.owner}/${context.repo.repo}`) {
    throw new Error('Only PRs merged into this repository\'s main can be backported.');
  }

  // Fetch current labels rather than relying on an old event payload on retries.
  const targets = targetBranches(pr.labels);
  for (const branch of targets) {
    try {
      await github.rest.repos.getBranch({ ...context.repo, branch });
    } catch (error) {
      if (error.status === 404) {
        throw new Error(`Target ${branch} does not exist. Create the release branch or remove its backport label.`);
      }
      throw error;
    }
  }
  core.setOutput('pull_number', String(pr.number));
  core.setOutput('targets', targets.join(' '));
  core.info(targets.length
    ? `Backport #${pr.number} to ${targets.join(', ')}`
    : `No backport v<major>.<minor> labels on #${pr.number}; nothing to do.`);
}

module.exports = { targetBranches, prepare };
