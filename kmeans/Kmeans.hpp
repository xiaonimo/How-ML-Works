#ifndef KMEANS_HPP
#define KMEANS_HPP
#include <ctime>
#include <vector>
#include <string>
#include <random>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "datatype.hpp"

class Kmeans {
public:
    Kmeans(const points_t& _data_set, unsigned int _n_clusters, points_t _centers=points_t(), std::string _Metric="L2", double _eps=0.01, bool _verbose=true):
          data_set(_data_set), data_dim(_data_set[0].size()), data_cnt(_data_set.size()), verbose(_verbose), Metric(_Metric), eps(_eps),
          n_clusters(_n_clusters), centers(_centers) {
        if (n_clusters > data_cnt) throw std::invalid_argument("n_clusters is too big!");
        if (_centers.empty()) _init_centers();
    }
    void cluster();

public:
    const points_t& data_set;
    const unsigned int data_dim;
    const unsigned int data_cnt;

    bool verbose;
    const std::string Metric;
    const double eps;
    const unsigned int n_clusters;
    std::vector<std::vector<double>> centers;
    std::vector<std::vector<std::size_t>> cluster_res;

private:
    void _cluster();
    void _init_centers();
    void _update_centers();
    index_t _get_nearest_center(const point_t&);
    double _metric(const point_t&, const point_t&);

private:
    double _sum_dist_cur=std::numeric_limits<double>::max();
    double _sum_dist_last=0.;
};

void Kmeans::cluster() {
    index_t _itr = 0;
    while (_itr==0 || (_itr<20 && _sum_dist_last-_sum_dist_cur>eps)) {
        auto t1 = clock();
        _cluster();
        auto t2 = clock();
        _update_centers();

        if (verbose) {
            std::cout <<"iteration:"<< _itr << " cur_dist:" << _sum_dist_last-_sum_dist_cur << " time:"<<(t2-t1)/double(CLOCKS_PER_SEC)<<std::endl;
            //for (auto v:cluster_res) std::cout << v.size() <<"\t";
            //std::cout << std::endl;
        }
        ++_itr;
    }
    if (verbose) std::cout << "cluster finished!" << std::endl;
}

void Kmeans::_init_centers() {
    std::mt19937 _gen;
    std::uniform_int_distribution<int> _random_index(0, data_cnt-1);

    std::unordered_map<int, bool> hmap;
    for (index_t i=0; i<n_clusters;) {
        index_t __index = _random_index(_gen);
        if (hmap[__index]) continue;
        hmap[__index] = true;
        centers.push_back(data_set[__index]);
        ++i;
    }
    if (verbose) std::cout << "init centers finished!" <<std::endl;
}

void Kmeans::_cluster() {
    //_sum_dist_last = _sum_dist_cur;
    //_sum_dist_cur = 0;
    cluster_res = std::vector<std::vector<index_t>>(n_clusters);

    for (index_t _index=0; _index<data_cnt; ++_index) {
        index_t _center_index = _get_nearest_center(data_set[_index]);
        cluster_res[_center_index].push_back(_index);
        //_sum_dist_cur += _metric(centers[_center_index], data_set[_index]);
    }
}

index_t Kmeans::_get_nearest_center(const point_t &p) {
    index_t _res = -1;
    double _min_dist = std::numeric_limits<double>::max();
    for (index_t _index=0; _index<n_clusters; ++_index) {
        double _cur_dist = _metric(p, centers[_index]);
        if (_cur_dist >= _min_dist) continue;
        _min_dist = _cur_dist;
        _res = _index;
    }
    return _res;
}

void Kmeans::_update_centers() {
    //每个聚类中心
    for (index_t c=0; c<n_clusters; ++c) {
        point_t _new_center(data_dim);
        index_t _cluster_sz = cluster_res[c].size();
        //每个簇中的点
        for (index_t p=0; p<_cluster_sz; ++p) {
            //每个维度
            for (index_t _dim=0; _dim<data_dim; ++_dim)
                _new_center[_dim] += data_set[cluster_res[c][p]][_dim]/double(_cluster_sz);
        }
        centers[c] = _new_center;
    }

    _sum_dist_last = _sum_dist_cur;
    _sum_dist_cur = 0.;
    for (index_t i=0; i<n_clusters; ++i) {
        for (index_t j=i+1; j<n_clusters; ++j) {
            _sum_dist_cur += _metric(centers[i], centers[j]);
        }
    }
}

double Kmeans::_metric(const point_t& p1, const point_t& p2) {
    double _dist = 0;
    //index_t _data_dim = p1.size();
    if (Metric == std::string("L2")) {
        for (index_t _dim=0; _dim<data_dim; ++_dim) {
            _dist += std::pow(p1[_dim]-p2[_dim], 2);
        }
    }
    return _dist;
}

#endif // KMEANS_HPP
