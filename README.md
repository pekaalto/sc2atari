#### Info & References

Here [deepmind's sc2 environment](https://github.com/deepmind/pysc2/) is simplified and converted
to [OpenAI's gym](https://github.com/openai/gym) environment  so that any existing atari-codes can be applied to simplified sc2-minigames.

The **FullyConv** -policy (smaller version) from https://deepmind.com/documents/110/sc2le.pdf is implemented and plugged into
[OpenAI-Baselines](https://github.com/openai/baselines) a2c implementation.

With this the 3 easiest mini-games can be "solved" quickly.

#### Results

<table align="center">
  <tr>
    <td align="center">Map</td>
    <td align="center">Episodes</td>
    <td align="center">Avg score</td>
    <td align="center">Max score</td>
    <td align="center">Deepmind avg</td>
    <td align="center">Deepmind max</td>
  </tr>
  <tr>
    <td align="center">MoveToBeacon</td>
    <td align="center">32*200</td>
    <td align="center">25</td>
    <td align="center">30</td>
    <td align="center">26</td>
    <td align="center">45</td>
  </tr>
  <tr>
    <td align="center">CollectMineralShards**</td>
    <td align="center">32*3100</td>
    <td align="center">66</td>
    <td align="center">94</td>
    <td align="center">103</td>
    <td align="center">134</td>
  </tr>
    <tr>
      <td align="center">DefeatRoaches**</td>
      <td align="center">48*1800</td>
      <td align="center">38</td>
      <td align="center">258</td>
      <td align="center">100</td>
      <td align="center">355</td>
    </tr>
</table>

**CollectMineralShards and DefeatRoaches performance was still improving slightly

- Avg and max are from the last n_envs*100 episodes.
- For all maps used the parameters seen in the repo except n_envs=32 (48 in DefeatRoaches).
- Episodes is the total number of playing-episodes over all environments.

Deepmind scores are shown for comparison.
They are the FullyConv ones reported in the [release paper](https://deepmind.com/documents/110/sc2le.pdf).

#### How to run
Install the requirements (Baselines etc) below, clone the repo and do

`python run_sc2_a2c.py --map_name MoveToBeacon --n_envs 32`

This won't save any files. Some results are printed to stdout.

#### Requirements
- Python 3 (will NOT work with python 2)
- [Open AI's baselines](https://github.com/openai/baselines) (tested with 0.1.4)
(Can also skip the installation and dump the baselines folder inside this repo, most of the dependencies in baselines are not really if use only a2c)
- [pysc2](https://github.com/deepmind/pysc2/) (tested with v1.2)
- Tensorflow (tested with 1.3.0)
- Other standard python packages like numpy etc.



#### Notes
Here we use only the **screen-player-relative** observation from the original observation space.
Action space is limited only to one action: **Select army followed by Attack Move** (same for the author when he plays sc2).

With this slice from observation/action space we can make agent to learn the 3 mini-games mentioned above.
However for anything more complicated it's not enough.

The action/obs-space limitation makes the problem very much easier, faster and less general/interesting.
Because of this and the differences in the network and hyperparamteres the scores are not directly comparable with the release-paper.

The achieved scores here are considerably lower than the Deepmind results
which suggests that the limited action space is not enough to achieve optimal performance (e.g micro against roaches or using two marines separately in shards).