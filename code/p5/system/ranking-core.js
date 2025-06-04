// rankingCore.js

class RankingSystem {
    constructor(paramList) {
        this.paramsList = paramList;
        this.n = paramList.length;
        this.comparisons = [];
        this.results = Array(this.n).fill(0);
        this.progressCount = 0;

        this.totalComparisons = 0;
        this.currentPair = [];

        this._initializeComparisons();
    }

    _initializeComparisons() {
        for (let i = 0; i < this.n; i++) {
            for (let j = i + 1; j < this.n; j++) {
                this.comparisons.push([i, j]);
            }
        }
        shuffle(this.comparisons, true);
        this.totalComparisons = this.comparisons.length;
    }

    hasNext() {
        return this.comparisons.length > 0;
    }

    nextPair() {
        this.currentPair = this.comparisons.pop();
        return this.currentPair;
    }

    vote(winnerSide) {
        const [i, j] = this.currentPair;
        const winner = winnerSide === "left" ? i : j;
        this.results[winner]++;
        this.progressCount++;
    }

    getProgress() {
        return {
            done: this.progressCount,
            total: this.totalComparisons,
            percentage: (this.progressCount / this.totalComparisons) * 100
        };
    }

    getResults() {
        return this.results;
    }

    getRankedIndices() {
        const scores = this.results.map((score, idx) => ({ idx, score }));
        scores.sort((a, b) => b.score - a.score);

        const ranking = Array(this.n).fill(0);
        let currentRank = 1;

        for (let i = 0; i < scores.length; i++) {
            if (i > 0 && scores[i].score < scores[i - 1].score) {
                currentRank = i + 1;
            }
            ranking[scores[i].idx] = currentRank;
        }

        return ranking;
      }
  }