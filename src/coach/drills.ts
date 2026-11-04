import type { Fault } from "../analysis/metrics"

export interface Drill {
  id: string
  name: string
  /** What it fixes, in a few words */
  fixes: string
  faults: Fault[]
  /** What you need */
  kit: string
  steps: string[]
  reps: string
  /** What it should feel like when it's working */
  feel: string
}

/** A free drill library. Each drill is a common, widely taught exercise, written out step by step. */
export const DRILLS: Drill[] = [
  {
    id: "count-tempo",
    name: "Count your tempo",
    fixes: "A rushed or slow rhythm",
    faults: ["tempo-quick", "tempo-slow"],
    kit: "Any club, or a metronome app",
    steps: [
      "Set a metronome to about 70 beats a minute, or count out loud.",
      "Start the backswing on beat 1 and reach the top on beat 3.",
      "Swing down so you reach the ball on the next beat, 4.",
      "Make five swings without a ball, then five with one, keeping the same count.",
    ],
    reps: "3 sets of 10 swings",
    feel: "Unhurried at the top, then one smooth push down.",
  },
  {
    id: "pause-at-top",
    name: "Pause at the top",
    fixes: "Rushing from the top",
    faults: ["tempo-quick"],
    kit: "A mid iron",
    steps: [
      "Make a normal backswing.",
      "Hold the top for a full second: say \"one\" out loud.",
      "Start down with your lower body, letting the arms follow.",
      "Hit half-speed shots first, then build up.",
    ],
    reps: "2 sets of 10 balls",
    feel: "The downswing starts from your feet and hips, not your hands.",
  },
  {
    id: "pump-drill",
    name: "Pump drill",
    fixes: "A slow, hesitant transition",
    faults: ["tempo-slow"],
    kit: "A mid iron",
    steps: [
      "Swing to the top.",
      "Start down until your hands reach hip height, then go back to the top.",
      "Do that pump once more, then swing all the way through.",
    ],
    reps: "10 swings",
    feel: "A continuous, flowing change of direction.",
  },
  {
    id: "wall-head",
    name: "Steady head against the wall",
    fixes: "Head moving sideways",
    faults: ["head-sway", "head-height"],
    kit: "A wall, a cushion, no club",
    steps: [
      "Take your golf posture with the top of your head lightly touching a cushion on a wall.",
      "Cross your arms on your chest and turn back and through.",
      "Keep the same light pressure on the cushion the whole time.",
    ],
    reps: "3 sets of 10 turns",
    feel: "Your body turns around your head, which stays put.",
  },
  {
    id: "eyes-on-spot",
    name: "Eyes on a spot",
    fixes: "Head rising or dipping",
    faults: ["head-height"],
    kit: "A tee or coin",
    steps: [
      "Put a coin a few centimetres in front of the ball.",
      "Keep looking at the coin through impact, until the ball is gone.",
      "Hold your head height; feel your chest turn under it.",
    ],
    reps: "15 balls",
    feel: "You see the coin after the ball has left.",
  },
  {
    id: "alignment-stick-hip",
    name: "Stick at the trail hip",
    fixes: "Hips sliding away in the backswing",
    faults: ["hip-sway"],
    kit: "An alignment stick or umbrella",
    steps: [
      "Push a stick into the ground just outside your trail hip, standing straight up.",
      "Make backswings without touching the stick.",
      "Turn your belt buckle away from the target instead of moving it sideways.",
    ],
    reps: "3 sets of 10 backswings, then 10 balls",
    feel: "Pressure into the inside of your trail foot, not its outside edge.",
  },
  {
    id: "feet-together",
    name: "Feet together",
    fixes: "Swaying and poor balance",
    faults: ["hip-sway", "head-sway", "finish-balance"],
    kit: "A short iron",
    steps: [
      "Stand with your feet touching.",
      "Hit easy three-quarter shots.",
      "If you lose balance, slow down until you can hold the finish.",
    ],
    reps: "15 balls",
    feel: "You have to turn, because sliding knocks you over.",
  },
  {
    id: "towel-arms",
    name: "Towel under the arms",
    fixes: "Arms and body out of sync",
    faults: ["lead-arm-bent", "short-turn"],
    kit: "A small towel",
    steps: [
      "Tuck a towel under both armpits.",
      "Make half swings without letting it drop.",
      "Let your chest turn carry the arms.",
    ],
    reps: "3 sets of 10 half swings",
    feel: "Arms and chest move together as one piece.",
  },
  {
    id: "lead-arm-only",
    name: "Lead arm only",
    fixes: "A bent lead arm at the top",
    faults: ["lead-arm-bent"],
    kit: "A short iron, gripped down",
    steps: [
      "Hold the club in your lead hand only, gripped down a few centimetres.",
      "Swing back to where your hands are at shoulder height, keeping the arm extended but not stiff.",
      "Swing through to the same height on the other side.",
    ],
    reps: "3 sets of 10 swings",
    feel: "A wide arc, with the club head far from your body.",
  },
  {
    id: "cross-arm-turn",
    name: "Club across the chest",
    fixes: "A short shoulder turn",
    faults: ["short-turn"],
    kit: "A club, no ball",
    steps: [
      "Hold a club across your chest with both arms crossed over it.",
      "Take your posture and turn back until the shaft points at the ball or past it.",
      "Turn through until it points at the target.",
    ],
    reps: "3 sets of 10 turns",
    feel: "Your back faces the target at the top.",
  },
  {
    id: "hold-finish",
    name: "Hold the finish",
    fixes: "Hanging back at the finish",
    faults: ["finish-balance"],
    kit: "Any club",
    steps: [
      "Hit a shot and hold your finish for three seconds.",
      "Check: belt buckle facing the target, trail heel up, weight on your lead foot.",
      "If you wobble, swing a little slower next time.",
    ],
    reps: "Every ball of a practice session",
    feel: "Standing tall and balanced on your lead leg.",
  },
  {
    id: "step-drill",
    name: "Step through",
    fixes: "Weight staying on the trail side",
    faults: ["finish-balance"],
    kit: "A short iron",
    steps: [
      "Start with your feet together.",
      "As you start the backswing, step your lead foot toward the target.",
      "Swing down as the foot lands, and finish on that foot.",
    ],
    reps: "10 balls",
    feel: "A natural shift toward the target before the club comes down.",
  },
  {
    id: "hips-to-wall",
    name: "Hips to the wall",
    fixes: "Early extension, standing up",
    faults: ["early-extension"],
    kit: "A wall or a chair",
    steps: [
      "Take your posture with your backside just touching a wall (or chair back) behind you.",
      "Turn back and keep your trail hip touching it.",
      "Turn through so your lead hip touches it, instead of moving away from it.",
    ],
    reps: "3 sets of 10 slow swings",
    feel: "Your hips go back and around, not toward the ball.",
  },
  {
    id: "setup-check",
    name: "Posture check",
    fixes: "Too upright or too hunched at address",
    faults: ["posture-upright", "posture-hunched", "knee-flex"],
    kit: "A club and a mirror or phone camera",
    steps: [
      "Stand tall holding the club against your thighs.",
      "Push your hips back until the club tip touches just above your knees, keeping your back straight.",
      "Let your knees soften slightly, then let your arms hang straight down and grip the club there.",
      "Check side-on in a mirror: back straight, tilted forward, arms hanging under your shoulders.",
    ],
    reps: "Before every few shots",
    feel: "Athletic, like a defender ready to move.",
  },
]

export const drillsFor = (fault: string) => DRILLS.filter((d) => d.faults.includes(fault as Fault))
export const drillById = (id: string) => DRILLS.find((d) => d.id === id)
