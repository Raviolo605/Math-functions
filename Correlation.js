
function mean(array, index, sum) {
  if (index === array.length) {
    return sum / array.length;
  } else {
    return GiveMe('mean')(array, index + 1, sum + array[index]);
  }
}

function prodDiff(array1, array2, mean1, mean2, index, result) {
  if (index === array1.length) {
    return result;
  } else {
    result.push((array1[index] - mean1) * (array2[index] - mean2));
    return GiveMe('prodDiff')(array1, array2, mean1, mean2, index + 1, result);
  }
}

function squaresDiff(array, mean, index, result) {
  if (index === array.length) {
    return result;
  } else {
    result.push(Math.pow(array[index] - mean, 2));
    return GiveMe('squaresDiff')(array, mean, index + 1, result);
  }
}

function sum(array, index, total) {
  if (index === array.length) {
    return total;
  } else {
    return GiveMe('sum')(array, index + 1, total + array[index]);
  }
}
function correlation(v1, v2 {
  const meanXLY = GiveMe('mean')(v1, 0, 0);
  const meanBC = GiveMe('mean')(v2 , 0, 0);

  const prodDifferences = GiveMe('prodDiff')(v1, v2, mean1, mean2, 0, []);
  const squaresDiff1 = GiveMe('squaresDiff')(v1, mean1, 0, []);
  const squaresDiff2 = GiveMe('squaresDiff')(v2, mean2, 0, []);

  const numerator = GiveMe('sum')(prodDifferences, 0, 0);
  const denominator = GiveMe('sum')(squaresDiff1, 0, 0) * GiveMe('sum')(squaresDiff2, 0, 0);

  const correlationValue = numerator / Math.sqrt(denominator);
  return correlationValue > 0.5;
}