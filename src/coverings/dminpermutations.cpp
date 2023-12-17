/*
 * Algorithm for obtaining the limit distribution of minimum-distance selection
 * criterion.
 *
 * Authors: Malte J. Ziebarth (ziebarth@gfz-potsdam.de)
 *
 * Copyright (C) 2023 Malte J. Ziebarth
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */


#include <chrono>
#include <thread>


#include <algorithm>
#include <stdexcept>
#include <coverings/dminpermutations.hpp>
#include <boost/geometry/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/point.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/core/coordinate_system.hpp>

namespace bg = boost::geometry;

class sample_exceedance_error : public std::exception {

};


typedef std::pair<double, std::vector<size_t>> sample_t;
typedef std::vector<sample_t> sample_vec_t;

struct node_t {
	std::vector<size_t> neighbors;

	node_t() : neighbors(0)
	{
	}
};

/*
 * Types that describe the node permutation, i.e. the order of and
 * selection availability / block status of a node permutation.
 * In both algorithms, the permutation is built "left to right", pushing
 * selected nodes onto the stack and keeping the information about
 * blocked nodes up to date.
 */
struct fork_t {
	size_t i;
	double ln_p;

	fork_t(size_t i, double ln_p) : i(i), ln_p(ln_p) {};

	fork_t() : i(-1), ln_p(std::nan("")) {}
};

struct node_status_t {
	bool blocked;
	size_t blocker;

	node_status_t() : blocked(false), blocker(-1) {}

	void block(size_t blckr){
		if (blocked)
			return;
		blocked = true;
		blocker = blckr;
	}
};



static std::pair<std::vector<node_t>, size_t>
compute_neighbors(const double* xy, const size_t N, const double dmin)
{
	std::pair<std::vector<node_t>, size_t> result(N, 0);
	std::vector<node_t>& nodes = result.first;

	const double dmin2 = dmin*dmin;
	if (N < 100){
		/*
		 * Simple quadratic loop over the pairs.
		 */
		for (size_t i=0; i<N; ++i){
			const double xi = xy[2*i];
			const double yi = xy[2*i+1];
			for (size_t j=i+1; j<N; ++j){
				const double dx = xi - xy[2*j];
				const double dy = yi - xy[2*j+1];
				if (dx*dx + dy*dy <= dmin2){
					nodes[i].neighbors.push_back(j);
					nodes[j].neighbors.push_back(i);
					++result.second;
				}
			}
		}
	} else {
		/* Use an RTree for asymptotic N*log(N) complexity for very big
		 * sample sizes.
		 * Note that this is worth it only if overlaps are sparse. Otherwise,
		 * the permutations will create a huge output vector in any case.
		 */
		typedef bg::model::d2::point_xy<double, bg::cs::cartesian> xy_t;
		typedef bg::model::box<xy_t> box_t;
		typedef std::pair<xy_t, size_t> value_t;
		bg::index::rtree<value_t, bg::index::linear<16>> rtree;
		for (size_t i=0; i<N; ++i){
			rtree.insert(std::pair<xy_t,size_t>(xy_t(xy[2*i], xy[2*i+1]), i));
		}
		std::vector<value_t> res;
		for (size_t i=0; i<N; ++i){
			xy_t xy_i(xy[2*i], xy[2*i+1]);
			box_t qbox(xy_t(xy_i.x()-dmin, xy_i.y()-dmin),
			           xy_t(xy_i.x()+dmin, xy_i.y()+dmin));
			auto within_dmin = [=](const value_t& v) -> bool {
				const double dx = v.first.x() - xy_i.x();
				const double dy = v.first.y() - xy_i.y();
				return dx*dx + dy*dy <= dmin2;
			};
			res.clear();
			rtree.query(bg::index::intersects(qbox)
			            && bg::index::satisfies(within_dmin),
			            std::back_inserter(res));
			for (const value_t& val : res){
				if (val.second == i)
					continue;
				nodes[i].neighbors.push_back(val.second);
				if (val.second > i)
					++result.second;
			}
		}
	}

	return result;
}


/*
 * Common code.
 *
 * The NodeStack class keeps track of permutation- and exclusion information
 * as a valid data choice is being built node by node.
 */



class NodeStack {
public:
	NodeStack(const std::vector<node_t>& nodes)
	   : stack(nodes.size()), status(nodes.size()), nodes(nodes), _blocked(0)
	{
		/* We initialized to the correct size, but we start with an empty
		 * stack:
		 */
		stack.resize(0);
	}

	void clear() {
		stack.clear();
		std::fill(status.begin(), status.end(), node_status_t());
		_blocked = 0;
	}

	bool empty() const {
		return stack.empty();
	}

	fork_t top() const {
		return stack.back();
	}

	size_t top_node() const {
		return stack.back().i;
	}

	double top_log_probability() const {
		return stack.back().ln_p;
	}

	size_t size() const {
		return stack.size();
	}

	size_t blocked() const {
		return _blocked;
	}

	bool is_blocked(size_t index) const {
		return status[index].blocked;
	}

	size_t free_index(size_t n = 0) const {
		size_t i=0;
		++n;
		for (; i<nodes.size(); ++i){
			if (!status[i].blocked){
				--n;
				if (n == 0)
					break;
			}
		}
		return i;
	}

	void push(size_t i){
		/*
		 * Start with the probability of the previous level (or 1, in case
		 * this is the first node):
		 */
		const double ln_p0 = (stack.empty()) ? 0.0 : stack.back().ln_p;

		/* Chance of selecting this node next: */
		const double ln_p = ln_p0 - std::log(nodes.size() - _blocked);

		/* Block the neighbors: */
		for (size_t n : nodes[i].neighbors){
			if (!status[n].blocked){
				status[n].block(i);
				++_blocked;
			}
		}

		/* Block the node by itself: */
		status[i].block(i);
		++_blocked;

		/* Add to the node stack: */
		stack.emplace_back(i, ln_p);
	}

	void pop() {
		size_t i = stack.back().i;
		stack.pop_back();
		/*
		 * Unblock blocked nodes:
		 */
		for (size_t n : nodes[i].neighbors){
			if (status[n].blocker == i){
				status[n] = node_status_t();
				--_blocked;
			}
		}
		status[i] = node_status_t();
		--_blocked;
	}

	std::vector<fork_t>::const_iterator cbegin() const {
		return stack.cbegin();
	}

	std::vector<fork_t>::const_iterator cend() const {
		return stack.cend();
	}

private:
	std::vector<fork_t> stack;
	std::vector<node_status_t> status;
	const std::vector<node_t>& nodes;
	size_t _blocked;
};


/*
 *
 * Methods to transfer the the current NodeStack selection to the data
 * structure that keeps track of the discovered samples (and their weight),
 * and to transfer from that working data structure to the sample vector
 * that gets returned.
 *
 */

static void
samples_add_sample(const NodeStack& stack,
                   std::map<std::vector<size_t>,double>& samples)
{
	/*
	 * Fill the node stack content to a simple vector, key of the map,
	 * and sort the contained nodes.
	 */
	std::vector<size_t> sample(stack.size());
	std::transform(stack.cbegin(), stack.cend(),
	               sample.begin(),
	               [](const fork_t& f) -> size_t
	               {
	                   return f.i;
	               });
	std::sort(sample.begin(), sample.end());

	/*
	 * Add the sample or increase the existing sample's probability:
	 */
	auto it = samples.lower_bound(sample);
	if (it == samples.end() || it->first != sample){
		samples.emplace_hint(it, std::move(sample), stack.top().ln_p);
	} else {
		/*
		 * Addition, formulated in logarithms.
		 *    it->second = ln(a)
		 *    stack.top().ln_p = ln(b)
		 * We want:
		 *    it->second = ln(a+b)
		 * which is
		 *    ln(a*(1+b/a)) = ln(a) + log1p(b/a) = ln(a) + log1p(exp(ln(b) - ln(a)))
		 *                  = it->second  +  log1p(exp( stack.top().ln_p  -  it->second ))
		 */
		it->second += std::log1p(std::exp(stack.top().ln_p - it->second));
	}
}


static sample_vec_t
samples_generate_result(std::map<std::vector<size_t>,double>& samples)
{
	sample_vec_t result(samples.size());
	auto rit = result.begin();
	for (size_t i=0; i<result.size(); ++i)
	{
		/* Transfer  */
		auto n = samples.extract(samples.begin());
		rit->first = n.mapped();
		rit->second.swap(n.key());
		++rit;
	}

	return result;
}



/*
 * Full deterministic sampling of the permutation space.
 *
 * This function will obtain all possible permutations that adhere to the
 * distance exclusion graph but it will take ages to do so (read: probably
 * factorial time dependency on an average data set). This method might not
 * be feasible for data sets larger than ~30-40 data points that are part of
 * the exclusion graph. Hence the requirement to specify `max_samples` and
 * `max_iter`.
 */


static sample_vec_t
dsr_deterministic(std::vector<node_t>& nodes, const size_t max_samples,
                  const size_t max_iter)
{
	/*
	 * Types:
	 */

	NodeStack node_stack(nodes);

	node_stack.push(0);

	std::map<std::vector<size_t>,double> samples;
	size_t iter = 0;
	while (samples.size() < max_samples && iter++ < max_iter
	       && !node_stack.empty())
	{
		/*
		 * Iteratively handle nodes:
		 */
		while (node_stack.blocked() < nodes.size()){
			/*
			 * Find a free node:
			 */
			size_t i = node_stack.free_index();
			if (i == nodes.size())
				throw std::runtime_error("Could not find free node although "
				                         "there should be.");
			node_stack.push(i);
		}

		/*
		 * Add the sample. Check if it's a duplicate. If so, add the
		 * probability mass to the existing sample.
		 */
		samples_add_sample(node_stack, samples);

		/*
		 * Increment the current level or move up one layer:
		 */
		while (!node_stack.empty()){
			/*
			 * Remember the previous node and remove it from the stack:
			 */
			size_t i = node_stack.top_node();
			node_stack.pop();

			/*
			 * Find the next free one:
			 */
			bool success = false;
			for (++i; i<nodes.size(); ++i){
				if (!node_stack.is_blocked(i)){
					node_stack.push(i);
					success = true;
					break;
				}
			}

			if (success){
				break;
			} else {
				/* Move up a layer. */
			}
		}
	}

	/* Check if the algorithm was successful: */
	if (!node_stack.empty()){
		throw sample_exceedance_error();
	}

	return samples_generate_result(samples);
}




/*
 * Monte-Carlo (or quasi-Monte-Carlo) sampling of the permutation space.
 *
 * This function generates random permutations of nodes, adhering to the
 * exclusion graph. Permutations are built node-after-node, always checking
 * the number of remaining options.
 */


static sample_vec_t
dsr_monte_carlo(std::vector<node_t>& nodes,
                std::function<size_t(size_t)>& sample_generator,
                const size_t max_samples, const size_t max_iter)
{
	NodeStack node_stack(nodes);
	std::map<std::vector<size_t>,double> samples;

	for (size_t k=0; k<max_iter; ++k){
		/*
		 * Exit if we have already determined as many samples as wanted:
		 */
		if (samples.size() == max_samples)
			break;

		/*
		 * Start node.
		 * Here we use a uniform sampling of the nodes. The sampling
		 * strategy is a mixture of linear iteration and pseudo-randomness:
		 * First, we start with one pass of iterating all nodes once
		 * (and selecting follow-up nodes pseudo-randomly). I.e. start
		 * node of iteration k is nodes[k] for k < nodes.size().
		 * This will ensure that our generated samples contain each node
		 * at least once. The reason why this is important is the
		 * determination of the global Qmax:
		 *     Qmax = min(q[i] / c[i])
		 * If we use fully pseudo-random sampling, there is a chance that
		 * the data pair (q[i],c[i]) with the lowest ratio is not selected
		 * and Qmax may thereby vary pseudo-randomly by execution. This is
		 * not really a desireable quality; we instead would like to prescribe
		 * the behavior that each data point matters.
		 *
		 * After the first n=nodes.size() iterations, we select the start
		 * node pseudo-randomly. For large max_iter, the convergence property
		 * of this algorithm should hence be pseudo-random. For small max_iter,
		 * it may be just a bit closer to low discrepancy due to the added
		 * regular structure (?).
		 */
		node_stack.clear();
		if (k < nodes.size())
			node_stack.push(k);
		else
			node_stack.push(sample_generator(nodes.size()));

		/*
		 * Iteratively add random nodes from the pool of free nodes:
		 */
		size_t free;
		while ((free = nodes.size() - node_stack.blocked())){
			/*
			 * Find a free node:
			 */
			size_t i = node_stack.free_index(sample_generator(free));
			if (i == nodes.size())
				throw std::runtime_error("Could not find free node although "
				                         "there should be.");
			node_stack.push(i);
		}

		/*
		 * Add the sample. Check if it's a duplicate. If so, add the
		 * probability mass to the existing sample.
		 */
		samples_add_sample(node_stack, samples);
	}

	return samples_generate_result(samples);
}



/*
 * The main routine that bundles all of the above.
 *
 *
 */


std::vector<std::pair<double, std::vector<size_t>>>
reheatfunq::determine_restricted_samples(
    const double* xy,
    const size_t N,
    const double dmin,
    const size_t max_samples,
    const size_t max_iter,
    std::shared_ptr<std::function<size_t(size_t)>> sample_generator,
    bool extra_debug_checks
)
{
	/*
	 * Step 1: Build the graph of mutual exclusion
	 * We use two different algorithms of different asymptotic complexity
	 * based on the sample size. It's not really worth setting up a distance
	 * query tree if N < ???.
	 */
	size_t link_count = 0;
	std::vector<node_t> nodes(0);
	{
		auto res = compute_neighbors(xy, N, dmin);
		std::swap(nodes, res.first);
		link_count = res.second;
	}

	/* If no link, early exit: */
	if (link_count == 0){
		/* Found no overlap. Return all nodes! */
		std::vector<std::pair<double, std::vector<size_t>>> samples(1);
		samples[0].first = 1.0;
		samples[0].second.resize(N);
		for (size_t i=0; i<N; ++i){
			samples[0].second[i] = i;
		}
		return samples;
	}

	/*
	 * Extract all nodes that have no neighbors into a separate list that
	 * is always added:
	 */
	std::vector<size_t> always_added;
	std::vector<size_t> n2i(0);
	{
		/*
		 * Add all nodes without neighbors to always_added and
		 * condense all remaining to the beginning of nodes:
		 */
		std::vector<size_t> i2n(N, -1);
		auto dest = nodes.begin();
		size_t j = 0;
		for (size_t i=0; i<N; ++i){
			if (nodes[i].neighbors.empty())
				always_added.push_back(i);
			else {
				*dest = nodes[i];
				++dest;
				/* Keep track of the new index to update all neighbors later: */
				i2n[i] = j;
				/* Index map from the packed beginning to the original node
				 * vector: */
				n2i.push_back(i);
				++j;
			}
		}
		/*
		 * Update the neighbors:
		 */
		for (size_t i=0; i<nodes.size(); ++i){
			for (size_t& k : nodes[i].neighbors){
				k = i2n[k];
			}
		}
	}

	/*
	 * Call algorithms:
	 */
	sample_vec_t samples(0);
	try {
		/*
		 * The number of permutations grows more or less factorial with
		 * number of nodes. We may therefore not be able to compute all of
		 * them if the number of nodes is large enough (even moderate sample
		 * sizes <100 might be too much).
		 * Try it first here, then on fail fall back to Monte-Carlo sampling
		 * (if a sampling point generator has been provided).
		 */
		samples = dsr_deterministic(nodes, max_samples, max_iter);
	} catch (const sample_exceedance_error&){
		if (sample_generator){
			samples = dsr_monte_carlo(nodes, *sample_generator, max_samples,
			                          max_iter);
		} else {
			throw std::runtime_error("More than the supplied upper limit of "
		                              "samples.");
		}
	}


	/*
	 * Check sortedness:
	 */
	if (extra_debug_checks){
		auto it = samples.cbegin();
		for (auto it2 = it+1; it2 < samples.cend(); ++it2){
			if (it->second == it2->second)
				throw std::runtime_error("Not sorted!");
			++it;
		}
	}

	/* Norm the sample (important for Monte-Carlo version).
	 * First compute the logarithm of the maximum likely element
	 * as a reference for the probabilities:
	 */
	long double max_log = -std::numeric_limits<long double>::infinity();
	for (const sample_t& s0 : samples){
		if (s0.first > max_log)
			max_log = s0.first;
	}
	/*
	 * Compute probabilities relative to this reference level:
	 */
	for (sample_t& s0 : samples){
		s0.first = std::exp(s0.first - max_log);
	}
	/* Now compute the norms: */
	long double norm = 0.0;
	for (const sample_t& s0 : samples){
		norm += s0.first;
	}
	if (norm == 0.0)
		throw std::runtime_error("determine_restricted_samples: Norm is zero.");
	if (norm < 0.0)
		throw std::runtime_error("determine_restricted_samples: Somehow, norm is negative.");
	for (sample_t& s0 : samples){
		s0.first /= norm;
	}

	/*
	 * Add the non-conflicting nodes and translate back to original indices:
	 */
	for (sample_t s : samples){
		for (size_t& i : s.second){
			i = n2i[i];
		}
		s.second.insert(s.second.end(), always_added.cbegin(),
		                always_added.cend());
		std::sort(s.second.begin(), s.second.end());
	}

	return samples;
}
