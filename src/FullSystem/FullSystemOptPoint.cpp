/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */
#include "math.h"
#include "stdio.h"
#include <algorithm>

#include <Eigen/LU>
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "util/globalFuncs.h"
#include "util/globalCalib.h"
#include "IOWrapper/ImageDisplay.h"

#include "FullSystem/FullSystem.h"
#include "FullSystem/ImmaturePoint.h"

namespace dso
{
	PointHessian* FullSystem::optimizeImmaturePoint(
		ImmaturePoint* point, int minObs,
		ImmaturePointTemporaryResidual* residuals)
	{
		int nres = 0;
		for (FrameHessian* fh : frameHessians)
		{
			if (fh != point->host)
			{
				residuals[nres].state_NewEnergy = residuals[nres].state_energy = 0;
				residuals[nres].state_NewState = ResState::OUTLIER;
				residuals[nres].state_state = ResState::INLIER;
				residuals[nres].target = fh;
				nres++;
			}
		}
		assert(nres == ((int)frameHessians.size()) - 1);

		bool print = false;//rand()%50==0;

		float lastEnergy = 0;
		float lastHdd = 0;
		float lastbd = 0;
		float currentIdepth = (point->idepth_max + point->idepth_min)*0.5f;

		for (int i = 0; i < nres; i++)
		{
			lastEnergy += point->linearizeResidual(&Hcalib, 1000, residuals + i, lastHdd, lastbd, currentIdepth);
			residuals[i].state_state = residuals[i].state_NewState;
			residuals[i].state_energy = residuals[i].state_NewEnergy;
		}

		// lastHdd表示关键点的Hessian也表示协方差的逆
		// 因此lastHdd越大表示当前待优化点的方差越小可信度越高
		if (!std::isfinite(lastEnergy) || lastHdd < setting_minIdepthH_act)
		{
			if (print)
				printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres, lastHdd, lastEnergy);
			return 0;
		}

		if (print)
			printf("Activate point. %d residuals. H=%f. Initial Energy: %f. Initial Id=%f\n",
				nres, lastHdd, lastEnergy, currentIdepth);

		float lambda = 0.1;
		for (int iteration = 0; iteration < setting_GNItsOnPointActivation; iteration++)
		{
			float H = lastHdd;
			H *= 1 + lambda;
			float step = (1.0 / H) * lastbd;
			float newIdepth = currentIdepth - step;

			float newHdd = 0; float newbd = 0; float newEnergy = 0;
			for (int i = 0; i < nres; i++)
				newEnergy += point->linearizeResidual(&Hcalib, 1, residuals + i, newHdd, newbd, newIdepth);

			if (!std::isfinite(lastEnergy) || newHdd < setting_minIdepthH_act)
			{
				if (print) printf("OptPoint: Not well-constrained (%d res, H=%.1f). E=%f. SKIP!\n",
					nres, newHdd, lastEnergy);
				return 0;
			}

			if (print)
				printf("%s %d (L %.2f) %s: %f -> %f (idepth %f)!\n",
				(true || newEnergy < lastEnergy) ? "ACCEPT" : "REJECT",
					iteration,
					log10(lambda),
					"",
					lastEnergy, newEnergy, newIdepth);

			if (newEnergy < lastEnergy)
			{
				currentIdepth = newIdepth;
				lastHdd = newHdd;
				lastbd = newbd;
				lastEnergy = newEnergy;
				for (int i = 0; i < nres; i++)
				{
					residuals[i].state_state = residuals[i].state_NewState;
					residuals[i].state_energy = residuals[i].state_NewEnergy;
				}

				lambda *= 0.5;
			}
			else
			{
				lambda *= 5;
			}

			// 迭代终止条件若得到的step已经很小了那么就不用再进行迭代了
			if (fabsf(step) < 0.0001*currentIdepth)
				break;
		}

		if (!std::isfinite(currentIdepth))
		{
			printf("MAJOR ERROR! point idepth is nan after initialization (%f).\n", currentIdepth);
			return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
		}

		int numGoodRes = 0;
		for (int i = 0; i < nres; i++)
			if (residuals[i].state_state == ResState::INLIER) numGoodRes++;

		// 优化后的点最少在一个关键帧中是好的
		if (numGoodRes < minObs)
		{
			if (print) printf("OptPoint: OUTLIER!\n");
			return (PointHessian*)((long)(-1));		// yeah I'm like 99% sure this is OK on 32bit systems.
		}

		PointHessian* p = new PointHessian(point, &Hcalib);
		if (!std::isfinite(p->energyTH)) { delete p; return (PointHessian*)((long)(-1)); }

		p->lastResiduals[0].first = 0;
		p->lastResiduals[0].second = ResState::OOB;
		p->lastResiduals[1].first = 0;
		p->lastResiduals[1].second = ResState::OOB;
		p->setIdepthZero(currentIdepth);
		p->setIdepth(currentIdepth);
		p->setPointStatus(PointHessian::ACTIVE);

		for (int i = 0; i < nres; i++)
			if (residuals[i].state_state == ResState::INLIER)
			{
				PointFrameResidual* r = new PointFrameResidual(p, p->host, residuals[i].target);
				r->state_NewEnergy = r->state_energy = 0;
				r->state_NewState = ResState::OUTLIER;
				r->setState(ResState::INLIER);
				p->residuals.emplace_back(r);

				if (r->target == frameHessians.back())
				{
					p->lastResiduals[0].first = r;
					p->lastResiduals[0].second = ResState::INLIER;
				}
				else if (r->target == (frameHessians.size() < 2 ? 0 : frameHessians[frameHessians.size() - 2]))
				{
					p->lastResiduals[1].first = r;
					p->lastResiduals[1].second = ResState::INLIER;
				}
			}

		if (print) printf("point activated!\n");

		statistics_numActivatedPoints++;
		return p;
	}
}