class Dataset {
  public:
    Histogram &histogram(const int i) {
      return Histogram(*this);
    }
  private:
    std::vector<double> m_binEdges;
    std::vector<double> m_values;
    std::vector<double> m_errors;

    friend class Histogram;
};


class Histogram {
  public:
    Histogram(const Histogram &other) {}
  private:
    const double *m_binEdges; // might be shared, can only be const (actually also better to avoid messing up data)
    double *m_values;
    double *m_errors;
    std::unique_ptr<std::vector<double>> m_data{nullptr};
    //gsl::index m_stride{1}; // TODO measure performance overhead of supporting stride
};

// Requirements:
// 1. Cheap to pass by value
// 2. Data should be held either directly or in Dataset?
//
// BinEdges always const!? (allows for sharing)
// Unit?!


// reference to dataset? pointer to dataset?
//

Dataset d1;
Dataset d2;
auto h1 = d1.histogram(i); // Dataset does not contain histograms, this returns by value, but h references d!? Copy constructor, assignment??
auto h2 = d2.histogram(i);
h1 *= 2.0; // scale Y and E in d1
h1 = h2; // compare X in d1 and d2, copy Y and E from d2 to d1
Histogram h;
h = h2; // not in any Dataset

// Same thing, but histograms stored in Dataset, in *addition* to X, Y, E?
Dataset d1;
Dataset d2;
auto &h1 = d1.histogram(i); // reference, h1 references data in d1
auto h2 = d1.histogram(i); // value, copy constructor extracts data, stored internally?
h1 = h2; // sets data in d1, works only if unit ok??



class Histogram {
  public:
    // accessors to X, Y, E
  private:
    gsl::index m_size;
    const UnitId m_unit; // not a reference, but always const, must be know at time of construction (probably want 2 o 3 fields?) how to square errors in-place? store init in X, Y, and E?
    const double &m_binEdges; // always const to support sharing?
    double &m_values;
    double &m_errors;
    std::unique_ptr<std::vector<double>> m_data{nullptr};
};

// Histogram convertStdDevToVariance(const Histogram &histogram);


// handle unit in Histogram, not in Values, Errors (to get arround threading issues)? What about BinEdges? will lead to out-of-date units if data columns in dataset are modified!
// just check the unit? handle as other values? reference unit in dataset, propagate for stand-alone histograms
// - how to modify dataset including uit change?
// h1 *= h2; // unit is global to dataset cannot use on histogram level!
// => use d1 *= d2? // no! may not want to edit all columns
// d1.get<Values, Errors> *= d2.get<Values, Errors>; // full column, can handle units
// use column in Dataset for units??
// unit as attribute of Histogram *column*?
// h1 *= h2; // fail, cannot edit unit?
// d1.get<Histograms> *= d2.get<Histograms>; // edits unit
// d1.get<Values> *= d2.get<Values>; // bypasses units??!! :( no! unit stored on BinEdges, Values, and Errors!
//
// Dataset d;
// d.addColumn(BinEdges{});
// d.addColumn(Values{});
// d.addColumn(Errors{});
// d.addHistogram(); // references BinEdges, Values, Errors
// d.addColumn(Units{});
//

Histogram myalg(const Histogram &hist1, const Histogram &hist2) {
  // hist1 *= hist2; // fail because it modifies unit?
  return hist1 * hist2; // creates new histogram with new unit! adds overhead because it prevents in-place operation? -> for more performance, operate with Dataset?
  // How do we put this back into a Dataset? Can it be done in-place? Copy data, ignoring units, set unit once? Don't do it in-place?
  // d1.get<Values>() *= d2.get<Values>(); // ?
  // If histogram referencing data in Dataset cannot be modifed unless unit is unchanged, is it still useful?
}

Dataset d;
d.get<ColumnId::BinEdges>(); // may have different dimension, not useful in combination with Values and Errors below!
d.get<ColumnId::Values>();
d.get<ColumnId::Errors>();

// How should we write rebin()?
Dataset rebin(const Dataset &d, const std::vector<BinEdge> &binEdges); // copies everything?
void rebin(Dataset &d, const Dataset &binEdges) {
  // 1. Add new data (value + error) column to d with same dimensions but different TOF length.
  // 2. Add new binEdges as new column
  // 3.
  for(auto &item : DatasetIterator<OldEdges, OldData, NewEdges, NewData>) { // how can we have several such items (old + new), unless we use strings to identify them?? should have names and Ids?
    rebin(item);
  }
  // 4. Remove old data and old binEdges.

}
// would rather write code at a lower level, not knowing about Dataset! Applying
// to Dataset should happen automatically!


d.apply(rebin)

// Advantages of Dataset:
// - single implementation of extracting/slicing, merging, chopping
// - single implementation for loading/saving and visualization?
// Questions:
// - Arbitrary columns?
//   - How to identify columns in this case? strings feels error prone, and cumbersome, since we also need to use the type: d.get<double>("counts");
//     Allow getting by type, throw if duplicate?!
//   - If not, aren't we too restrictive, e.g., for tables? (could use extra dimension if multiple columns of same type are needed??)
// - should we store Values and Errors as separate columns, or a single column if std::pair<Value, Error>? Is there data without errors?


void apply(Dataset &d, const std::function &f);
// Use signature of f to determine which columns to apply to and which dimensions are core dimensions?